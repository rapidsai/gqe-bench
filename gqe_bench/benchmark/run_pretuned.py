#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import argparse
import os
import re
import sqlite3
import sys

from database_benchmarking_tools.experiment import ExperimentDB
from database_benchmarking_tools.utility import generate_db_path
from gqe_bench import lib
from gqe_bench.benchmark.gqe_experiment import GqeExperimentConnection
from gqe_bench.benchmark.run import (
    CatalogContext,
    DataInfo,
    QueryExecutionContext,
    QueryInfoContext,
    boost_shared_memory_pool_size,
    identifier_type_to_sql,
    parse_bool,
    parse_scale_factor,
    run_suite,
    set_eager_module_loading,
    setup_db,
    sql_to_identifier_type,
)
from gqe_bench.param_sweep_config import (
    BENCHMARK_CONFIG_DEFAULTS,
    get_validation_dir,
)


def get_best_parameters_file(sqlite_file: str):
    conn = sqlite3.connect(sqlite_file)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM gqe_best_parameters")
    best_parameters = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return best_parameters


def get_best_parameters_folder(df_folder: str):
    best_param_dict = {}
    for df_file in os.listdir(df_folder):
        db_path = os.path.join(df_folder, df_file)
        if not os.path.isfile(db_path):
            continue
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT * FROM gqe_best_parameters")
            rows = cursor.fetchall()
            for row in rows:
                q_name = row["q_name"]
                r_avg_duration_s = row["r_avg_duration_s"]
                # If this q_name is not seen yet, or this row is better, keep it
                if (
                    q_name not in best_param_dict
                    or r_avg_duration_s < best_param_dict[q_name]["r_avg_duration_s"]
                ):
                    best_param_dict[q_name] = dict(row)
        except Exception as e:
            print(f"Skipping {db_path} because of error getting best parameters: {e}")
            pass
        finally:
            conn.close()

    # Extract query number from names like "Q1", "Q16_opt", etc.
    def extract_query_num(q_name):
        # Remove "Q" prefix and extract digits only
        match = re.search(r"Q(\d+)", q_name)
        return int(match.group(1)) if match else 0

    return sorted(best_param_dict.values(), key=lambda x: extract_query_num(x["q_name"]))


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "dataset",
        help="Dataset location. Can be a TPC-H dataset or a custom dataset. For custom dataset a DDL file is required.",
    )
    arg_parser.add_argument("plan", help="Substrait query plan location")
    arg_parser.add_argument("solution", help="Reference results location with pattern")
    arg_parser.add_argument(
        "--swept-sqlite-file",
        help="SQLite file that has the parameter sweep results",
        type=str,
        default=None,
    )
    arg_parser.add_argument(
        "--swept-sqlite-folder", help="Folder that has the parameter sweep results"
    )
    arg_parser.add_argument("--output", "-o", help="Output file path")
    arg_parser.add_argument("--output-physical-plan", help="Output file folder of physical plans")
    arg_parser.add_argument(
        "--queries",
        "-q",
        help="Which queries to run",
        nargs="+",
        action="extend",
        type=str,
    )
    arg_parser.add_argument(
        "--load-all-data",
        "-l",
        help="Whether to load all data or the data required for each query. Defaults to 0",
        type=int,
        choices=[0, 1],
        default=0,
    )
    # CUPTI support hasn't been added for multi-process yet. Mark these options as mutually exclusive.
    metrics_group = arg_parser.add_mutually_exclusive_group()
    metrics_group.add_argument(
        "--metrics",
        help='Profile CUPTI range metrics. Examples: "pcie__read_bytes.sum", "pcie__write_bytes.sum"',
        nargs="+",
        action="extend",
        type=str,
        default=None,
    )
    metrics_group.add_argument(
        "--multiprocess", "-m", help="Run in multiprocess mode", action="store_true"
    )
    arg_parser.add_argument(
        "--repeat", "-rep", help="How many times to run each query", type=int, default=6
    )
    arg_parser.add_argument(
        "--boost_pool_size", help="Boost pool size in GBs", type=int, default=None
    )
    arg_parser.add_argument(
        "--validate-results",
        help="Validate results before writing the timing entries to the database. Defaults to True.",
        type=parse_bool,
        default=True,
    )
    arg_parser.add_argument(
        "--validate-dir",
        help="Scratch directory to write query results to for validation. Defaults to a temporary directory via tempfile.",
        type=str,
        default=BENCHMARK_CONFIG_DEFAULTS["validate_dir"],
    )
    arg_parser.add_argument(
        "--ddl-file-path",
        help="Path to DDL file",
        type=str,
        default=BENCHMARK_CONFIG_DEFAULTS["ddl_file_path"],
    )
    arg_parser.add_argument(
        "--suite-name",
        help="Suite name (e.g., TPC-H, TPC-DS). Defaults to TPC-H.",
        type=str,
        default=BENCHMARK_CONFIG_DEFAULTS["suite_name"],
    )
    args = arg_parser.parse_args()
    # TODO: add --nsys-trace to collect nsys traces for the best parameters

    repeat = args.repeat
    gqe_host = "localhost"
    scale_factor = parse_scale_factor(args.dataset)

    set_eager_module_loading()

    physical_plan_folder = args.output_physical_plan if args.output_physical_plan else None

    errors = []

    if args.swept_sqlite_file:
        best_parameters = get_best_parameters_file(args.swept_sqlite_file)
    elif args.swept_sqlite_folder:
        best_parameters = get_best_parameters_folder(args.swept_sqlite_folder)
    else:
        raise ValueError("Either --swept_sqlite_file or --swept_sqlite_folder must be specified")

    validate_dir = get_validation_dir(args.validate_dir)

    # TODO: Multiprocess mode needs to check if spawned ranks is equal to that in the best parameters
    # https://gitlab-master.nvidia.com/haog/gqe-python/-/issues/13
    multiprocess_runtime_context = None
    if args.multiprocess:
        lib.mpi_init()

        all_storage_kind_is = lambda params, kind: all(
            bp.get("d_storage_device_kind") == kind for bp in params
        )

        all_storage_kind = best_parameters[0].get("d_storage_device_kind")
        if all_storage_kind_is(best_parameters, "boost_shared_memory"):
            pool_size = boost_shared_memory_pool_size(
                args.boost_pool_size, scale_factor, args.load_all_data
            )
            print(f"Initializing CPU shared memory with pool size {pool_size}")
            lib.initialize_shared_memory(pool_size)
        elif not all_storage_kind_is(best_parameters, "parquet_file"):
            raise ValueError(
                "Multiprocess mode is only supported with parquet_file storage kind or boost_shared_memory storage kind"
            )
        multiprocess_runtime_context = lib.MultiProcessRuntimeContext(
            lib.scheduler_type.ROUND_ROBIN, all_storage_kind
        )

    is_root_rank = (not args.multiprocess) or (lib.mpi_rank() == 0)
    edb_file = None
    edb_config = None
    if is_root_rank:
        edb_file = (
            args.output if args.output else generate_db_path(f"gqe", args.suite_name, gqe_host)
        )
        edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(GqeExperimentConnection)
        edb_config.create_experiment_db()
        print(f"Writing SQLite file to {edb_file}")

    if physical_plan_folder and is_root_rank:
        print(f"Writing Physical Plan to the folder {physical_plan_folder}")

    def run_all(edb_info):
        errors_local = []
        invalid_results_local = []
        for best_parameter in best_parameters:
            query_str = best_parameter["q_name"].lstrip("Q")
            if best_parameter["q_source"] == "custom_substrait":
                query_idx = -1
            else:
                query_idx = int(query_str.split("_")[0])

            if args.queries and query_str not in args.queries:
                print(
                    f"Skipping {best_parameter['q_name']} because it is not in the list of queries to run"
                )
                continue

            num_row_groups = best_parameter["de_num_row_groups"]
            char_type = best_parameter["d_char_type"]
            use_opt_type_for_single_char_col = char_type == "char"
            compression_format = best_parameter["de_compression_format"]
            compression_ratio_threshold = best_parameter["de_compression_ratio_threshold"]
            secondary_compression_format = best_parameter["de_secondary_compression_format"]
            secondary_compression_ratio_threshold = best_parameter[
                "de_secondary_compression_ratio_threshold"
            ]
            secondary_compression_multiplier_threshold = best_parameter[
                "de_secondary_compression_multiplier_threshold"
            ]
            use_cpu_compression = best_parameter["de_use_cpu_compression"]
            compression_level = best_parameter["de_compression_level"]
            compression_chunk_size = best_parameter["de_compression_chunk_size"]
            zone_map_partition_size = best_parameter["de_zone_map_partition_size"]

            identifier_type = sql_to_identifier_type(best_parameter["d_identifier_type"])

            storage_kind = best_parameter["d_storage_device_kind"]
            query_source = best_parameter["q_source"]

            if best_parameter["d_scale_factor"] != scale_factor:
                print(
                    f"Skipping {best_parameter['q_name']} because scale factor {best_parameter['d_scale_factor']} does not match input scale factor {scale_factor}"
                )
                continue

            data_info = DataInfo(
                storage_device_kind=storage_kind,
                format="internal",
                location=None,  # FIXME: set location as NUMA node, iff set in GQE
                not_null=False,
                identifier_type=identifier_type_to_sql(identifier_type),
                char_type=char_type,
                decimal_type="float",
                scale_factor=scale_factor,
                num_row_groups=num_row_groups,
                compression_format=compression_format,
                compression_ratio_threshold=compression_ratio_threshold,
                secondary_compression_format=secondary_compression_format,
                secondary_compression_ratio_threshold=secondary_compression_ratio_threshold,
                secondary_compression_multiplier_threshold=secondary_compression_multiplier_threshold,
                use_cpu_compression=use_cpu_compression,
                compression_level=compression_level,
                compression_chunk_size=compression_chunk_size,
                zone_map_partition_size=zone_map_partition_size,
            )

            reference_file = args.solution.replace("%d", f"q{query_idx}")
            if query_source == "handcoded":
                substrait_file = None
            elif query_source == "substrait":
                substrait_file = os.path.join(args.plan, f"df_q{query_idx}.bin")
            elif query_source == "custom_substrait":
                substrait_file = os.path.join(args.plan, f"{query_str}.bin")
                reference_file = args.solution.replace("%d", f"{query_str}")
            else:
                raise ValueError(f"Invalid query source: {query_source}")

            query_info_ctx = QueryInfoContext(
                query_idx,
                query_str,
                query_source,
                reference_file,
                scale_factor,
                substrait_file,
                physical_plan_folder,
            )

            gqe_parameter = QueryExecutionContext(
                best_parameter["p_num_workers"],
                best_parameter["p_num_partitions"],
                best_parameter["p_use_overlap_mtx"],
                best_parameter["p_join_use_hash_map_cache"],
                best_parameter["p_read_use_zero_copy"],
                best_parameter["p_join_use_unique_keys"],
                best_parameter["p_join_use_perfect_hash"],
                best_parameter["p_join_use_mark_join"],
                best_parameter["p_use_partition_pruning"],
                best_parameter["p_filter_use_like_shift_and"],
                best_parameter["p_aggregation_use_perfect_hash"],
                query_info_ctx=query_info_ctx,
            )

            if best_parameter["d_scale_factor"] != scale_factor:
                print(
                    f"Skipping {best_parameter['q_name']} because scale factor {best_parameter['d_scale_factor']} does not match input scale factor {scale_factor}"
                )
                continue

            if query_source == "custom_substrait" and not args.load_all_data:
                print(
                    f"Skipping {best_parameter['q_name']} because custom substrait queries must be run with load_all_data=1"
                )
                continue

            cat_ctx = CatalogContext(
                args.dataset,
                storage_kind,
                num_row_groups,
                0 if args.load_all_data else -1,
                identifier_type,
                use_opt_type_for_single_char_col,
                args.ddl_file_path,
                zone_map_partition_size,
                compression_format,
                compression_chunk_size,
                compression_ratio_threshold,
                secondary_compression_format,
                secondary_compression_ratio_threshold,
                secondary_compression_multiplier_threshold,
                use_cpu_compression,
                compression_level,
            )

            run_suite(
                cat_ctx,
                data_info,
                scale_factor,
                [gqe_parameter],
                edb_file,
                edb_info,
                args.metrics,
                errors_local,
                invalid_results_local,
                repeat,
                is_root_rank,
                args.multiprocess,
                multiprocess_runtime_context,
                args.validate_results,
                validate_dir,
                args.suite_name,
            )

        return errors_local, invalid_results_local

    if is_root_rank:
        with edb_config as edb:
            edb_info = setup_db(edb)
        errors, invalid_results = run_all(edb_info)

        print(f"Finished SQLite file at {edb_file}")
        if physical_plan_folder:
            print(f"Finished Physical Plan at the folder {physical_plan_folder}")

        if invalid_results:
            print("The following configurations run successfully but produce incorrect results")
            for result in invalid_results:
                print(result)

        if errors:
            print("The following configurations produced errors:")
            for error in errors:
                print(error)
    else:
        errors, invalid_results = run_all(None)

    if args.multiprocess:
        multiprocess_runtime_context.finalize()
        if all_storage_kind == "boost_shared_memory" and lib.mpi_rank() == 0:
            lib.finalize_shared_memory()
        lib.mpi_finalize()

    if errors or invalid_results:
        sys.exit(1)


if __name__ == "__main__":
    main()
