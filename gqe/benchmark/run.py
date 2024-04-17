# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import Context
import gqe.lib
from gqe.benchmark.verify import verify_parquet
from gqe.relation import Relation

import os
import sqlite3
import pynvml
import socket
import platform
from dataclasses import dataclass
from typing import Union  # Not needed with Python>=3.10


@dataclass
class QueryInfo:
    identifier: str
    root_relation: Union[Relation, gqe.lib.Relation]
    reference_solution: str


@dataclass
class Parameter:
    num_partitions: int
    read_use_zero_copy: bool
    max_num_workers: int


def run_tpc(perf_db_file: str, catalog, queries: list[QueryInfo], parameters: list[Parameter]):
    repeat = 6

    out_conn = sqlite3.connect(perf_db_file)
    out_cursor = out_conn.cursor()

    print(f"Writing SQLite file to {perf_db_file}")

    measurements_schema = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "create_experiment_db.sql")

    with open(measurements_schema, 'r') as m_schema_handle:
        m_schema_sql = m_schema_handle.read()
        out_cursor.executescript(m_schema_sql)

    hw_info_id = insert_hw_info(out_cursor)

    errors = []

    for parameter in parameters:
        num_partitions = parameter.num_partitions
        read_use_zero_copy = parameter.read_use_zero_copy
        max_num_workers = parameter.max_num_workers
        join_use_hash_map_cache = bool(os.getenv("GQE_JOIN_USE_HASH_MAP_CACHE", False))

        print(f"Running with parameters num_partitions={num_partitions}, "
              f"read_use_zero_copy={read_use_zero_copy}, "
              f"num_workers={max_num_workers}")

        parameters_id = insert_parameters(
            out_cursor,
            {
                "num_workers": max_num_workers,
                "num_partitions": num_partitions,
                "join_use_hash_map_cache": join_use_hash_map_cache,
                "read_use_zero_copy": read_use_zero_copy,
            })

        # TODO: use with statement instead?
        context = Context(max_num_workers, num_partitions, read_use_zero_copy)

        for query in queries:
            print(f"Running {query.identifier}...")

            experiment_id = insert_experiment(
                out_cursor,
                {
                    "parameters_id": parameters_id,
                    "hw_info_id": hw_info_id,
                    "build_info_id": None,
                    "name": query.identifier,
                    "suite": "TPC-H",
                    "scale_factor": None,
                })

            for count in range(repeat):
                out_file = f"{query.identifier}_out.parquet"

                try:
                    elapsed_time = context.execute(catalog, query.root_relation, out_file)
                except Exception as error:
                    print(error)
                    break

                try:
                    verify_parquet(out_file, query.reference_solution)
                except AssertionError as error:
                    print(error)
                    errors.append((query.identifier, parameter))
                    break

                insert_run(
                    out_cursor,
                    {
                        "experiment_id": experiment_id,
                        "number": count,
                        "nvtx_marker": None,
                        "duration_s": elapsed_time / 1000,
                    })

        del context

    print("The following configurations run successfully but produce incorrect results")
    print(errors)

    out_conn.commit()
    print(f"Finished SQLite file at {perf_db_file}")


class GpuInfo:
    def __init__(self):
        pynvml.nvmlInit()

    def __del__(self):
        pynvml.nvmlShutdown()

    def cuda_driver_version(self):
        version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        return str(version // 1000) + "." + str((version // 10) % 10)

    def device_product_name(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        return pynvml.nvmlDeviceGetName(handle)

    def gpu_cores(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        return pynvml.nvmlDeviceGetNumGpuCores(handle)

    def max_memory_clock(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)

    def max_sm_clock(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)

    def pcie_link_generation(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        return pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)

    def system_driver_version(self):
        return pynvml.nvmlSystemGetDriverVersion()

    def total_ecc_errors(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        try:
            corrected = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, pynvml.NVML_AGGREGATE_ECC)
            uncorrected = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, pynvml.NVML_AGGREGATE_ECC)
            return corrected + uncorrected
        except pynvml.nvml.NVMLError_NotSupported:
            return None


def insert_hw_info(cursor):
    gpu_id = 0
    gpu_info = GpuInfo()

    entry = (
        socket.gethostname(),
        platform.machine(),
        gpu_info.device_product_name(gpu_id),
        gpu_info.system_driver_version(),
        gpu_info.cuda_driver_version(),
        gpu_info.pcie_link_generation(gpu_id),
        gpu_info.gpu_cores(gpu_id),
        gpu_info.max_sm_clock(gpu_id),
        gpu_info.max_memory_clock(gpu_id),
        gpu_info.total_ecc_errors(gpu_id)
    )
    print(entry)

    # Insert the HW info.
    #
    # There is a UNIQUE constraint, so we IGNORE if the row already exists.
    cursor.execute(
        " \
        INSERT OR IGNORE INTO hw_info( \
        h_hostname, \
        h_cpu_arch, \
        h_gpu_product_name, \
        h_nvidia_driver_version, \
        h_cuda_version, \
        h_gpu_pcie_link_generation, \
        h_gpu_cuda_cores, \
        h_gpu_max_clock_sm_mhz, \
        h_gpu_max_clock_memory_mhz, \
        h_gpu_ecc_errors \
        ) \
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
        ",
        entry
    )

    # Retrieve the primary key of the HW info row.
    #
    # We can't use `lastrowid`, because the INSERT OR IGNORE might not insert a
    # new row. In this case, `lastrowid` doesn't contain to the row we want.
    #
    # OR NULL is necessary to match rows that don't have a NOT NULL constraint.
    # FIXME: This isn't entirely correct, because theoretically it might return
    # multiple rows.
    cursor.execute(
        " \
        SELECT h_id \
        FROM hw_info \
        WHERE h_hostname = ? \
        AND h_cpu_arch = ? \
        AND NULL OR h_gpu_product_name = ? \
        AND NULL OR h_nvidia_driver_version = ? \
        AND NULL OR h_cuda_version = ? \
        AND NULL OR h_gpu_pcie_link_generation = ? \
        AND NULL OR h_gpu_cuda_cores = ? \
        AND NULL OR h_gpu_max_clock_sm_mhz = ? \
        AND NULL OR h_gpu_max_clock_memory_mhz = ? \
        AND NULL OR h_gpu_ecc_errors = ? \
        ",
        entry
    )

    return cursor.fetchone()[0]


def insert_parameters(cursor, entry):
    cursor.execute(
        " \
        INSERT OR IGNORE INTO parameters( \
        p_num_workers, \
        p_num_partitions, \
        p_join_use_hash_map_cache,  \
        p_read_use_zero_copy \
        ) \
        VALUES (:num_workers, :num_partitions, :join_use_hash_map_cache, :read_use_zero_copy) \
        ",
        entry
    )

    cursor.execute(
        " \
        SELECT p_id \
        FROM parameters \
        WHERE p_num_workers = :num_workers \
        AND p_num_partitions = :num_partitions \
        AND p_join_use_hash_map_cache = :join_use_hash_map_cache \
        AND p_read_use_zero_copy = :read_use_zero_copy \
        ",
        entry
    )

    return cursor.fetchone()[0]


def insert_experiment(cursor, entry):
    cursor.execute(
        " \
        INSERT INTO experiment( \
        e_parameters_id, \
        e_hw_info_id, \
        e_build_info_id, \
        e_name, \
        e_suite, \
        e_scale_factor \
        ) \
        VALUES (:parameters_id, :hw_info_id, :build_info_id, :name, :suite, :scale_factor) \
        RETURNING e_id \
        ",
        entry
    )

    return cursor.fetchone()[0]


def insert_run(cursor, entry):
    cursor.execute(
        " \
        INSERT INTO run( \
        r_experiment_id, \
        r_number, \
        r_nvtx_marker, \
        r_duration_s \
        ) \
        VALUES (:experiment_id, :number, :nvtx_marker, :duration_s) \
        RETURNING r_id \
        ",
        entry
    )

    return cursor.fetchone()[0]


def analyze(cursor):
    cursor.execute("ANALYZE")
