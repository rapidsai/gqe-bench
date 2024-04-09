# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import Catalog, execute
import argparse
import importlib
import sqlite3
import socket
import pynvml
import platform
import os
import time


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("location", help="TPC-H dataset location")
    args = arg_parser.parse_args()

    measurements_schema = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "create_experiment_db.sql")
    scale_factor = 100  # FIXME: add data set paths here to make SF configurable
    repeat = 6

    out_file = f'gqe_tpch_{ socket.gethostname() }.db3'
    out_conn = sqlite3.connect(out_file)
    out_cursor = out_conn.cursor()

    print(f"Writing SQLite file to {out_file}")

    with open(measurements_schema, 'r') as m_schema_handle:
        m_schema_sql = m_schema_handle.read()
        out_cursor.executescript(m_schema_sql)

    hw_info_id = insert_hw_info(out_cursor)

    parameters_id = insert_parameters(
        out_cursor,
        {
            "num_workers": os.getenv("MAX_NUM_WORKERS", 1),
            "num_partitions": os.getenv("MAX_NUM_PARTITIONS", 8),
            "join_use_hash_map_cache": os.getenv("GQE_JOIN_USE_HASH_MAP_CACHE", False),
            "read_use_zero_copy": os.getenv("GQE_READ_USE_ZERO_COPY", True),
        })

    catalog = Catalog()
    catalog.register_tpch(args.location, "memory")

    query_identifiers = ["tpch_q6", "tpch_q15", "tpch_q17", "tpch_q18", "tpch_q21"]

    for query_identifier in query_identifiers:
        print(f"Running {query_identifier}...")
        module = importlib.import_module(query_identifier)
        query = getattr(module, query_identifier)()

        experiment_id = insert_experiment(
            out_cursor,
            {
                "parameters_id": parameters_id,
                "hw_info_id": hw_info_id,
                "build_info_id": None,
                "name": query_identifier_to_name(query_identifier),
                "suite": "TPC-H",
                "scale_factor": scale_factor,
            })

        for count in range(repeat):
            start_time = time.time()
            execute(catalog, query.root_relation(), f"{query_identifier}_out.parquet")
            elapsed_time = time.time() - start_time

            insert_run(
                out_cursor,
                {
                    "experiment_id": experiment_id,
                    "number": count,
                    "nvtx_marker": None,
                    "duration_s": elapsed_time,
                })

    out_conn.commit()
    print(f"Finished SQLite file at {out_file}")


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


def query_identifier_to_name(identifier):
    # Convert "tpch_q6" -> "Q6"
    return identifier[5:].upper()


if __name__ == "__main__":
    main()
