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
from gqe.benchmark.gqe_experiment import GqeParameters
from gqe.relation import Relation

from database_benchmarking_tools import experiment as exp

import importlib.resources
import os
import nvtx
import re
from dataclasses import dataclass


@dataclass
class EdbInfo:
    sut_info_id: int
    hw_info_id: int
    query_source: str


@dataclass
class QueryInfo:
    identifier: str
    root_relation: Relation | gqe.lib.Relation
    reference_solution: str


@dataclass
class Parameter:
    num_partitions: int
    read_use_zero_copy: bool
    max_num_workers: int


def setup_db(edb: exp.ExperimentDB, query_source: str) -> EdbInfo:
    sut_creation_path = importlib.resources.files("gqe.benchmark").joinpath(
        "system_under_test.sql"
    )
    with importlib.resources.as_file(sut_creation_path) as script:
        edb.execute_script(script)

    sut_info_id = edb.insert_sut_info(exp.SutInfo(name="gqe"))
    hw_info_id = edb.insert_hw_info()

    return EdbInfo(sut_info_id, hw_info_id, query_source)


def parse_scale_factor(path: str) -> int:
    predicate = re.compile(".*(?:sf|SF)([0-9]+)([kK]?).*")
    matches = predicate.match(path)

    if matches is None:
        return None

    scale_factor = int(matches.group(1))
    if matches.group(2):
        scale_factor = scale_factor * 1000

    return scale_factor


# Note: Presumably needs to be set before CUDA initialization. CUDA docs don't
# mention when the variable needs to be set. Typically, users will set it
# before launching the program. In our case that's not possible, because the
# program is already running.
def set_eager_module_loading():
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"


def run_tpc(
    catalog,
    query: QueryInfo,
    scale_factor: int,
    parameters: list[Parameter],
    edb: exp.ExperimentDB,
    edb_info: EdbInfo,
    errors: list,
):
    repeat = 6

    for parameter in parameters:
        num_partitions = parameter.num_partitions
        read_use_zero_copy = parameter.read_use_zero_copy
        max_num_workers = parameter.max_num_workers
        join_use_hash_map_cache = bool(os.getenv("GQE_JOIN_USE_HASH_MAP_CACHE", False))
        debug_mem_usage = bool(os.getenv("GQE_PYTHON_DEBUG_MEM_USAGE", False))

        print(
            f"Running with parameters num_partitions={num_partitions}, "
            f"read_use_zero_copy={read_use_zero_copy}, "
            f"num_workers={max_num_workers}, "
            f"debug_mem_usage={debug_mem_usage}"
        )

        parameters_id = edb.insert_gqe_parameters(
            GqeParameters(
                sut_info_id=edb_info.sut_info_id,
                num_workers=max_num_workers,
                num_partitions=num_partitions,
                join_use_hash_map_cache=join_use_hash_map_cache,
                read_use_zero_copy=read_use_zero_copy,
            )
        )

        # TODO: use with statement instead?
        context = Context(max_num_workers, num_partitions, read_use_zero_copy, debug_mem_usage)

        print(f"Running {query.identifier}...")

        experiment_id = edb.insert_experiment(
            exp.Experiment(
                sut_info_id=edb_info.sut_info_id,
                parameters_id=parameters_id,
                hw_info_id=edb_info.hw_info_id,
                build_info_id=None,
                name=query.identifier,
                suite="TPC-H",
                scale_factor=scale_factor,
                query_source=edb_info.query_source,
            )
        )

        for count in range(repeat):
            out_file = f"{query.identifier}_out.parquet"

            with nvtx.annotate(f"Run {query.identifier}"):
                try:
                    elapsed_time = context.execute(
                        catalog, query.root_relation, out_file
                    )
                except Exception as error:
                    print(error)
                    break

            try:
                verify_parquet(out_file, query.reference_solution)
            except AssertionError as error:
                print(error)
                errors.append((query.identifier, parameter))
                break

            edb.insert_run(
                exp.Run(
                    experiment_id=experiment_id,
                    number=count,
                    nvtx_marker=None,
                    duration_s=elapsed_time / 1000,
                )
            )

        del context
