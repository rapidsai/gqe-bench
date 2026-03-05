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
import functools
import inspect
import itertools
import json
import os
import sys

from database_benchmarking_tools.experiment import ExperimentDB
from database_benchmarking_tools.utility import generate_db_path
from gqe_bench import lib
from gqe_bench.benchmark.gqe_experiment import GqeExperimentConnection
from gqe_bench.benchmark.run import (
    CatalogContext,
    DataInfo,
    QueryError,  # noqa: F401 needed because pickle for sandboxing
    QueryExecutionContext,
    QueryInfoContext,
    identifier_type_to_sql,
    is_valid_identifier_type,
    parse_bool,
    parse_identifier_type,
    parse_scale_factor,
    print_mp,
    run_suite,
    set_eager_module_loading,
    setup_db,
)
from gqe_bench.benchmark.sandboxing import run_sandboxed
from gqe_bench.param_sweep_config import (
    BENCHMARK_CONFIG_DEFAULTS,
    QUERY_CONFIG_DEFAULTS,
    check_cli_overrides,
    config_to_args,
    get_query_execution_params,
    get_validation_dir,
    load_json_config,
)


def get_queries(query_source: str, queries: list[str] = None, plan: str = None):
    # this is only for TPC-H queries
    tpch_handcoded_queries = [
        "1",
        "2",
        "2_fused_filter",
        "3",
        "3_fused_filter",
        "4",
        "5",
        "6",
        "7",
        "7_fused_filter",
        "9",
        "10",
        "10_fused_filter",
        "10_opt",
        "11",
        "12",
        "13",
        "13_opt",
        "13_fused",
        "15",
        "16",
        "16_opt",
        "17",
        "18",
        "18_opt",
        "19",
        "20",
        "20_fused_filter",
        "21",
        # Disabled because CI fails for SF0.01 [clutz, 2025-12-23]
        # "21_fused_filter",
        "21_opt",
        "22",
        "22_opt",
    ]

    # this is only for TPC-H queries
    # 1, 15, 18, 21 removed as per previous run-script
    # Q11 FIXME: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/141
    # Q4 removed as the substrait plan fails for SF1k
    tpch_substrait_queries = [
        "2",
        "3",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "12",
        "13",
        "14",
        "16",
        "17",
        "19",
        "20",
        "22",
    ]

    if query_source == "custom_substrait":
        custom_substrait_queries = [
            f.split(".bin")[0] for f in os.listdir(plan) if os.path.isfile(os.path.join(plan, f))
        ]
    else:
        custom_substrait_queries = []

    if queries:
        tpch_handcoded_queries = sorted(set(tpch_handcoded_queries) & set(queries))
        tpch_substrait_queries = sorted(set(tpch_substrait_queries) & set(queries))
        custom_substrait_queries = sorted(set(custom_substrait_queries) & set(queries))

    if query_source == "handcoded":
        if len(tpch_handcoded_queries) == 0:
            raise ValueError(f"No TPC-H handcoded queries found for the given queries")
        tpch_substrait_queries = []
        custom_substrait_queries = []
    elif query_source == "substrait":
        if len(tpch_substrait_queries) == 0:
            raise ValueError(f"No TPC-H substrait queries found for the given queries")
        tpch_handcoded_queries = []
        custom_substrait_queries = []
    elif query_source == "both":
        pass
    elif query_source == "custom_substrait":
        if len(custom_substrait_queries) == 0:
            raise ValueError(f"No custom substrait queries found for the given queries")
        tpch_handcoded_queries = []
        tpch_substrait_queries = []

    else:
        raise ValueError(f"Invalid query source: {query_source}")

    return tpch_handcoded_queries, tpch_substrait_queries, custom_substrait_queries


def get_custom_substrait_queries(plan):
    return [f.split(".bin")[0] for f in os.listdir(plan) if os.path.isfile(os.path.join(plan, f))]


# Helper method used to specify arguments that can take multiple boolean values
def add_boolean_list_argument(self, *args, **kwargs):
    """Specify an argument which takes a combination of true/false"""
    if "nargs" in kwargs or "type" in kwargs:
        raise argparse.ArgumentError(
            'Overwriting parameters "nargs" or "type" inside a boolean list argument definition'
        )
    self.add_argument(*args, **kwargs, type=parse_bool, nargs="+")


# Helper method used to specify arguments that can take multiple int values
def add_int_list_argument(self, *args, **kwargs):
    """Specify an argument which takes a list of int values"""
    if "nargs" in kwargs or "type" in kwargs:
        raise argparse.ArgumentError(
            'Overwriting parameters "nargs" or "type" inside a int list argument definition'
        )
    self.add_argument(*args, **kwargs, type=int, nargs="+")


def add_float_list_argument(self, *args, **kwargs):
    """Specify an argument which takes a list of float values"""
    if "nargs" in kwargs or "type" in kwargs:
        raise argparse.ArgumentError(
            'Overwriting parameters "nargs" or "type" inside a float list argument definition'
        )
    self.add_argument(*args, **kwargs, type=float, nargs="+")


def parse_args():
    arg_parser = argparse.ArgumentParser()

    # Monkey patch arg_parser to simplify definition of int and boolean arguments
    arg_parser.add_boolean_list_argument = functools.partial(add_boolean_list_argument, arg_parser)
    arg_parser.add_int_list_argument = functools.partial(add_int_list_argument, arg_parser)

    arg_parser.add_float_list_argument = functools.partial(add_float_list_argument, arg_parser)

    # JSON config file (when provided, other args are ignored)
    arg_parser.add_argument(
        "--json",
        "-j",
        help="Path to JSON configuration file. When provided, all other CLI arguments are ignored.",
        type=str,
        default=None,
    )

    # Positional arguments are optional when using --json
    arg_parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset location. Can be a TPC-H dataset or a custom dataset. For custom dataset a DDL file is required.",
    )
    arg_parser.add_argument("plan", nargs="?", help="Substrait query plan location")
    arg_parser.add_argument("solution", nargs="?", help="Reference results location with pattern")
    arg_parser.add_argument("--output", "-o", help="Output file path")
    arg_parser.add_argument("--quiet", help="Quiet mode", action="store_true")
    arg_parser.add_argument(
        "--queries",
        "-q",
        help="Which queries to run",
        nargs="+",
        action="extend",
        type=str,
    )
    arg_parser.add_argument(
        "--identifier-type",
        "-i",
        help="Identifier type used in the dataset. By default, a suitable type is automatically selected.",
        choices=["int32", "int64"],
        nargs="+",
        action="extend",
        type=str,
    )
    arg_parser.add_int_list_argument(
        "--row-groups",
        "-r",
        help="Number of row groups to use",
        action="extend",
    )
    arg_parser.add_int_list_argument(
        "--partitions",
        "-p",
        help="Number of partitions to use",
        default=QUERY_CONFIG_DEFAULTS["partitions"],
    )
    arg_parser.add_int_list_argument(
        "--workers",
        "-w",
        help="Number of workers to use",
        default=QUERY_CONFIG_DEFAULTS["workers"],
    )
    arg_parser.add_argument(
        "--query-source",
        help="Query source",
        choices=["handcoded", "substrait", "both", "custom_substrait"],
        default=BENCHMARK_CONFIG_DEFAULTS["query_source"],
    )
    arg_parser.add_argument(
        "--suite-name",
        help="Suite name (e.g., TPC-H, TPC-DS). Defaults to TPC-H.",
        type=str,
        default=BENCHMARK_CONFIG_DEFAULTS["suite_name"],
    )
    arg_parser.add_argument(
        "--compression-format",
        "-c",
        help="Compression format to use",
        choices=[
            "none",
            "ans",
            "lz4",
            "snappy",
            "gdeflate",
            "deflate",
            "cascaded",
            "zstd",
            "gzip",
            "bitcomp",
            "best_compression_ratio",
            "best_decompression_speed",
        ],
        nargs="+",
        default=BENCHMARK_CONFIG_DEFAULTS["compression_format"],
    )
    arg_parser.add_argument(
        "--load-all-data",
        "-l",
        help="Whether to load all data at once (1) or per query (0). If not specified, defaults to 1 for scale_factor <= 200, 0 otherwise",
        type=int,
        choices=[0, 1],
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
    arg_parser.add_argument(
        "--storage-kind",
        "-k",
        help="Storage kind",
        choices=[
            "pinned_memory",
            "system_memory",
            "numa_memory",
            "device_memory",
            "managed_memory",
            "numa_pinned_memory",
            "boost_shared_memory",
            "numa_pool_memory",
            "parquet_file",
        ],
        nargs="+",
        default=BENCHMARK_CONFIG_DEFAULTS["storage_kind"],
    )
    metrics_group.add_argument(
        "--num_ranks",
        "-nr",
        type=int,
        help=f"Set the number of GPUs to use. More than 1 GPU implies MPI/multiprocessing. Defaults to {BENCHMARK_CONFIG_DEFAULTS['num_ranks']} ",
        default=BENCHMARK_CONFIG_DEFAULTS["num_ranks"],
    )
    arg_parser.add_argument(
        "--repeat",
        "-rep",
        help="How many times to run each query",
        type=int,
        default=BENCHMARK_CONFIG_DEFAULTS["repeat"],
    )
    arg_parser.add_argument(
        "--query-timeout",
        "-qt",
        help=f"Timeout in (s) for queries in subprocesses. Used to kill suspected hanging jobs after timeout exceeded. Always used with -m. Ignored unless -sb and/or -m are set. Default: {BENCHMARK_CONFIG_DEFAULTS['query_timeout']}s",
        type=int,
        default=BENCHMARK_CONFIG_DEFAULTS["query_timeout"],
    )
    arg_parser.add_argument(
        "--data-timeout",
        "-dt",
        help=f"Timeout in (s) for data load subprocesses. Used to kill suspected hanging jobs after timeout exceeded. Always used with -m. Ignored unless -sb and/or -m are set. Default: {BENCHMARK_CONFIG_DEFAULTS['data_timeout']}s",
        type=int,
        default=BENCHMARK_CONFIG_DEFAULTS["data_timeout"],
    )
    arg_parser.add_argument(
        "--boost_pool_size", help="Boost pool size in GBs", type=int, default=None
    )
    arg_parser.add_argument(
        "--sandboxing",
        "-sb",
        help="Run with sandboxing. If process fails, a new process will start to run remaining queries.",
        action="store_true",
    )
    arg_parser.add_argument(
        "--validate-results",
        help="Validate results before writing the timing entries to the database. Defaults to True.",
        type=parse_bool,
        default=BENCHMARK_CONFIG_DEFAULTS["validate_results"],
    )
    arg_parser.add_argument(
        "--validate-dir",
        help="Scratch directory to write query results to for validation. Defaults to a temporary directory via tempfile.",
        type=str,
        default=BENCHMARK_CONFIG_DEFAULTS["validate_dir"],
    )

    # Arguments to enable/disable functionality
    arg_parser.add_boolean_list_argument(
        "--read-use-filter-pruning",
        "--partition-pruning",
        "-spp",
        help="Enable filter pruning (requires clustered dataset)",
        default=QUERY_CONFIG_DEFAULTS["read_use_filter_pruning"],
    )
    arg_parser.add_boolean_list_argument(
        "--read-use-overlap-mtx",
        "--use-overlap-mtx",
        help="Enable overlap mutex for read task",
        default=QUERY_CONFIG_DEFAULTS["read_use_overlap_mtx"],
    )
    arg_parser.add_boolean_list_argument(
        "--read-use-zero-copy",
        help="Enable zero-copy reads",
        default=QUERY_CONFIG_DEFAULTS["read_use_zero_copy"],
    )
    arg_parser.add_boolean_list_argument(
        "--filter-use-like-shift-and",
        help="Enable shift/and for LIKE expressions",
        default=QUERY_CONFIG_DEFAULTS["filter_use_like_shift_and"],
    )
    arg_parser.add_boolean_list_argument(
        "--join-use-hash-map-cache",
        help="Cache the join build-side hash map for multiple partitions",
        default=QUERY_CONFIG_DEFAULTS["join_use_hash_map_cache"],
    )
    arg_parser.add_boolean_list_argument(
        "--join-use-unique-keys",
        help="Enable unique key joins",
        default=QUERY_CONFIG_DEFAULTS["join_use_unique_keys"],
    )
    arg_parser.add_boolean_list_argument(
        "--join-use-perfect-hash",
        help="Enable perfect hashing for joins",
        default=QUERY_CONFIG_DEFAULTS["join_use_perfect_hash"],
    )
    arg_parser.add_boolean_list_argument(
        "--join-use-mark-join",
        help="Enable mark join",
        default=QUERY_CONFIG_DEFAULTS["join_use_mark_join"],
    )
    arg_parser.add_boolean_list_argument(
        "--aggregation-use-perfect-hash",
        help="Enable perfect hashing for aggregations",
        default=QUERY_CONFIG_DEFAULTS["aggregation_use_perfect_hash"],
    )
    arg_parser.add_float_list_argument(
        "--compression-ratio-threshold",
        help="Compression ratio threshold",
        default=BENCHMARK_CONFIG_DEFAULTS["compression_ratio_threshold"],
    )

    arg_parser.add_argument(
        "--secondary-compression-format",
        "-sc",
        help="Compression format to use",
        choices=[
            "none",
            "ans",
            "lz4",
            "snappy",
            "gdeflate",
            "deflate",
            "cascaded",
            "zstd",
            "gzip",
            "bitcomp",
        ],
        nargs="+",
        default=BENCHMARK_CONFIG_DEFAULTS["secondary_compression_format"],
    )

    arg_parser.add_float_list_argument(
        "--secondary-compression-ratio-threshold",
        help="Secondary compression ratio threshold",
        default=BENCHMARK_CONFIG_DEFAULTS["secondary_compression_ratio_threshold"],
    )

    arg_parser.add_float_list_argument(
        "--secondary-compression-multiplier-threshold",
        help="Secondary compression multiplier threshold",
        default=BENCHMARK_CONFIG_DEFAULTS["secondary_compression_multiplier_threshold"],
    )

    arg_parser.add_boolean_list_argument(
        "--use-cpu-compression",
        help="(Experimental; no effect in public build.) Use CPU compression for in-memory table compression",
        default=BENCHMARK_CONFIG_DEFAULTS["use_cpu_compression"],
    )

    arg_parser.add_int_list_argument(
        "--compression-level",
        help="Compression level (1-12). Higher values provide better compression but slower speed. Currently only supported for LZ4.",
        default=BENCHMARK_CONFIG_DEFAULTS["compression_level"],
    )
    arg_parser.add_int_list_argument(
        "--compression-chunk-size",
        help="Compression chunk size in bytes. Default: 128KB (2**17).",
        default=BENCHMARK_CONFIG_DEFAULTS["compression_chunk_size"],
    )
    arg_parser.add_int_list_argument(
        "--zone-map-partition-size",
        help="Zone map partition size. Default: 200000.",
        default=BENCHMARK_CONFIG_DEFAULTS["zone_map_partition_size"],
    )
    arg_parser.add_argument(
        "--ddl-file-path",
        help="Path to DDL file",
        type=str,
        default=BENCHMARK_CONFIG_DEFAULTS["ddl_file_path"],
    )

    return arg_parser.parse_args()


def main():
    args = parse_args()

    # Handle JSON configuration file
    if args.json:
        # Warn if other CLI args were provided
        check_cli_overrides(sys.argv)

        # Load JSON config and convert to args namespace
        json_path = args.json
        try:
            config = load_json_config(json_path)
            args = config_to_args(config)
            print(f"Loaded configuration from {json_path}")
        except FileNotFoundError:
            print(f"Error: Config file not found: {json_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error: Invalid config: {e}")
            sys.exit(1)
    else:
        # Validate required positional arguments when not using JSON
        if not args.dataset or not args.plan or not args.solution:
            print("Error: dataset, plan, and solution are required when not using --json")
            sys.exit(1)

    print(f"Arguments: {args}")

    if args.query_source == "custom_substrait" and not args.load_all_data:
        raise ValueError("Custom substrait queries must be run with load_all_data=1")
    if "parquet_file" in args.storage_kind and not args.load_all_data:
        raise ValueError("Parquet file queries must be run with load_all_data=1")

    if args.num_ranks < 1:
        print("# GPUs must be a positive integer")
        sys.exit(1)
    if args.query_timeout < 0 or args.data_timeout < 0:
        print(f"Timeouts cannot be negative: {args.query_timeout}, {args.data_timeout}")
        print("Timeout must be a positive integer or 0")
        print("Exiting with error")
        sys.exit(1)

    validate_dir = get_validation_dir(args.validate_dir)

    gqe_host = "localhost"
    scale_factor = parse_scale_factor(args.dataset)
    load_all_data = (
        args.load_all_data if args.load_all_data is not None else (1 if scale_factor <= 200 else 0)
    )
    # this is always true now that mpi has been deferred until later, but still needed as argument
    multiprocess_runtime_context = None
    if args.num_ranks > 1:
        if args.storage_kind != ["parquet_file"] and args.storage_kind != ["boost_shared_memory"]:
            raise ValueError(
                "Multiprocess mode is only supported with parquet_file storage kind or boost_shared_memory storage kind"
            )

    # If DDL file path is provided, identifier type is unused, we just default to int64
    # As DDL will specify the datatype for each column
    if args.ddl_file_path:
        identifier_type = [lib.TypeId.int64]
    else:
        if not args.identifier_type:
            identifier_type = [parse_identifier_type(args.dataset)]
        else:
            # You can set it to int32 or int64, for SF1k int64 is required.
            str_to_type = {"int32": lib.TypeId.int32, "int64": lib.TypeId.int64}
            identifier_type = [str_to_type[t] for t in args.identifier_type]

    set_eager_module_loading()

    edb_config = None
    edb_info = None
    edb_file = args.output if args.output else generate_db_path(f"gqe", args.suite_name, gqe_host)

    edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(GqeExperimentConnection)
    edb_config.create_experiment_db()
    edb_info = None
    with edb_config as edb:
        edb_info = setup_db(edb)
    print(f"Writing SQLite file to {edb_file}")

    handcoded_queries, substrait_queries, custom_substrait_queries = get_queries(
        args.query_source, args.queries, args.plan
    )

    def run_sweep(edb_file, edb_info, args):
        nonlocal identifier_type
        num_row_group_list = []
        if args.row_groups:
            num_row_group_list = args.row_groups
        elif scale_factor < 500:
            num_row_group_list = [1, 8]
        elif scale_factor < 1000:
            num_row_group_list = [4, 8]
        else:
            num_row_group_list = [16, 32]

        all_invalid_results = []
        all_errors = []
        for (
            num_row_groups,
            use_opt_type_for_single_char_col,
            compression_format,
            compression_ratio_threshold,
            secondary_compression_format,
            secondary_compression_ratio_threshold,
            secondary_compression_multiplier_threshold,
            use_cpu_compression,
            compression_level,
            compression_chunk_size,
            identifier_type,
            storage_kind,
            zone_map_partition_size,
        ) in itertools.product(
            num_row_group_list,
            [True],
            args.compression_format,
            args.compression_ratio_threshold,
            args.secondary_compression_format,
            args.secondary_compression_ratio_threshold,
            args.secondary_compression_multiplier_threshold,
            args.use_cpu_compression,
            args.compression_level,
            args.compression_chunk_size,
            identifier_type,
            args.storage_kind,
            args.zone_map_partition_size,
        ):
            # If a DDL file is not provided, we need to validate the identifier type
            # Else DDL file will provide the identifier type
            if not args.ddl_file_path:
                match is_valid_identifier_type(identifier_type, args.suite_name, scale_factor):
                    case True:
                        pass
                    case False:
                        continue
                    case None:
                        raise ValueError(
                            f"Unknown if identifier type {identifier_type} is valid for the given dataset"
                        )

            data_info = DataInfo(
                storage_device_kind=storage_kind,
                format="internal",
                location=None,  # FIXME: set location as NUMA node, iff set in GQE
                not_null=False,
                identifier_type=identifier_type_to_sql(identifier_type),
                char_type="char" if use_opt_type_for_single_char_col else "text",
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

            if load_all_data or (storage_kind == "parquet_file"):
                cat_ctx = CatalogContext(
                    args.dataset,
                    storage_kind,
                    num_row_groups,
                    0,
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
            else:
                # Pass in non-zero value to indicate load_all_data=False
                cat_ctx = CatalogContext(
                    args.dataset,
                    storage_kind,
                    num_row_groups,
                    -1,
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

            # We use a regular list to build so that we can sort it later before
            # moving the data into the list proxy.
            parameters = []
            for query_source, queries in [
                ("handcoded", handcoded_queries),
                ("substrait", substrait_queries),
                ("custom_substrait", custom_substrait_queries),
            ]:
                for query_str in queries:
                    # TODO: clean if else
                    # The custom queries might not have numbers in the name, so we use -1 as the query index
                    if query_source == "custom_substrait":
                        query_idx = -1
                        reference_file = args.solution.replace("%d", f"{query_str}")
                        substrait_file = os.path.join(args.plan, f"{query_str}.bin")
                    else:
                        query_idx = int(query_str.split("_")[0])
                        reference_file = args.solution.replace("%d", f"q{query_idx}")
                        substrait_file = None
                        if query_source == "substrait":
                            substrait_file = os.path.join(args.plan, f"df_q{query_idx}.bin")
                    physical_plan_folder = None

                    query_info_ctx = QueryInfoContext(
                        query_idx,
                        query_str,
                        query_source,
                        reference_file,
                        scale_factor,
                        substrait_file,
                        physical_plan_folder,
                    )

                    # Get query-specific parameters (merges JSON overrides with args defaults)
                    qp = get_query_execution_params(args, query_str)

                    for (
                        num_workers,
                        num_partitions,
                        use_overlap_mtx,
                        join_use_hash_map_cache,
                        read_use_zero_copy,
                        join_use_unique_keys,
                        join_use_perfect_hash,
                        join_use_mark_join,
                        use_partition_pruning,
                        filter_use_like_shift_and,
                        aggregation_use_perfect_hash,
                    ) in itertools.product(
                        qp["workers"],
                        qp["partitions"],
                        qp["read_use_overlap_mtx"],
                        qp["join_use_hash_map_cache"],
                        qp["read_use_zero_copy"],
                        qp["join_use_unique_keys"],
                        qp["join_use_perfect_hash"],
                        qp["join_use_mark_join"],
                        qp["read_use_filter_pruning"],
                        qp["filter_use_like_shift_and"],
                        qp["aggregation_use_perfect_hash"],
                    ):
                        # Skip zero copy for partition-row-group combinations where zero copy is not supported.
                        if read_use_zero_copy and (num_partitions != num_row_groups):
                            print_mp(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, num_partitions: {num_partitions}, num_row_groups: {num_row_groups} because zero copy requires row groups to be equal to partitions",
                                not args.quiet,
                            )
                            continue

                        # Zero copy can only be used for in-memory tables.
                        if read_use_zero_copy and storage_kind == "parquet_file":
                            print_mp(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, storage_kind: {storage_kind} because zero copy is not supported for parquet files",
                                not args.quiet,
                            )
                            continue

                        # Skip zero copy for compression, as compression takes precedence over zero copy in GQE read task.
                        #
                        # Note: Revisit if GQE behavior changes in future.
                        if read_use_zero_copy and compression_format != "none":
                            print_mp(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, compression_format: {compression_format} because zero copy is not supported with compression",
                                not args.quiet,
                            )
                            continue

                        if num_workers > num_partitions:
                            print_mp(
                                f"Skipping num_workers: {num_workers}, num_partitions: {num_partitions} because num_workers greater than num_partitions",
                                not args.quiet,
                            )
                            continue

                        if compression_format != "none" and use_overlap_mtx:
                            print_mp(
                                f"Skipping compression_format: {compression_format}, use_overlap_mtx: {use_overlap_mtx} because its not optimal to use overlap matrix with compression",
                                not args.quiet,
                            )
                            continue

                        # We want to always use hash map caching except when perfect hashing is used
                        # Because perfect join doesn't support hash map cache, see: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/161
                        if join_use_perfect_hash == join_use_hash_map_cache:
                            print_mp(
                                f"Skipping join_use_perfect_hash: {join_use_perfect_hash}, join_use_hash_map_cache: {join_use_hash_map_cache} because hash map caching should always be used except with perfect hashing",
                                not args.quiet,
                            )
                            continue

                        # Perfect hash join is disabled for substrait plans, see: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/161
                        if (query_source == "substrait" or query_source == "custom_substrait") and (
                            join_use_perfect_hash or aggregation_use_perfect_hash
                        ):
                            print_mp(
                                f"Skipping join_use_perfect_hash: {join_use_perfect_hash}, aggregation_use_perfect_hash: {aggregation_use_perfect_hash}, query_source: {query_source}. Because, perfect hash is only manually enabled in physical plans",
                                not args.quiet,
                            )
                            continue

                        # Skip parameter configuration where scale factor is so large that small partition/row group counts run into cuDF type limitations
                        # FIXME: Temp fix until partition redesign is complete: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/221
                        # Q11 and Q20 do not exhibit this issue and perform better with fewer partitions, so they are exceptional in this condition
                        if (
                            scale_factor > 500
                            and num_partitions < 4
                            and query_idx != 20
                            and query_idx != 11
                            and query_source != "custom_substrait"
                        ):
                            print_mp(
                                f"Skipping num_partitions: {num_partitions}, num_row_groups: {num_row_groups}, scale_factor: {scale_factor} because configuration is likely to encounter cuDF concatenate error",
                                not args.quiet,
                            )
                            continue

                        parameters.append(
                            QueryExecutionContext(
                                num_workers,
                                num_partitions,
                                use_overlap_mtx,
                                join_use_hash_map_cache,
                                read_use_zero_copy,
                                join_use_unique_keys,
                                join_use_perfect_hash,
                                join_use_mark_join,
                                use_partition_pruning,
                                filter_use_like_shift_and,
                                aggregation_use_perfect_hash,
                                query_info_ctx=query_info_ctx,
                            )
                        )
            # group same-query sets together to minimize data reloads
            # For custom queries, we could sort by query_str, but anyways we require the entire dataset to be loaded so the sort order doesn't impact loading time
            parameters.sort(key=lambda p: p.query_info_ctx.query_idx)

            is_mp = args.num_ranks > 1
            # Subprocess/Multi-GPU Case
            # Ideal case: spawn process, run to completion, join.
            # If subprocess fails, we iterate and launch the process again w/ the remaining parameters.
            if is_mp or args.sandboxing:
                # bind args so mpi process can call run_suite exactly like this
                func_sig = inspect.signature(run_suite)
                run_suite_args = func_sig.bind(
                    cat_ctx,
                    data_info,
                    scale_factor,
                    parameters,
                    edb_file,
                    edb_info,
                    args.metrics,
                    all_errors,
                    all_invalid_results,
                    args.repeat,
                    is_mp,
                    multiprocess_runtime_context,
                    args.validate_results,
                    validate_dir,
                    args.suite_name,
                    args.quiet,
                )
                run_suite_args.apply_defaults()
                run_sandboxed(
                    run_suite_args,
                    parameters,
                    load_all_data,
                    scale_factor,
                    all_errors,
                    all_invalid_results,
                    args,
                )
            else:
                print_mp(
                    "Running single-gpu experiments without subprocess sandbox",
                    not args.quiet,
                )
                run_suite(
                    cat_ctx,
                    data_info,
                    scale_factor,
                    parameters,
                    edb_file,
                    edb_info,
                    args.metrics,
                    all_errors,
                    all_invalid_results,
                    args.repeat,
                    is_mp,
                    multiprocess_runtime_context,
                    args.validate_results,
                    validate_dir,
                    args.suite_name,
                    args.quiet,
                )

        # Return should match outer loop indentation
        return all_errors, all_invalid_results

    errors, invalid_results = run_sweep(edb_file, edb_info, args)
    print(f"Finished SQLite file at {edb_file}")

    if invalid_results:
        print(
            f"The following {len(invalid_results)} configurations run successfully but produce incorrect results",
        )
        for result in invalid_results:
            print(result, args.quiet)

    if errors:
        print(f"The following {len(errors)} configurations produced errors")
        print("note: hard crashes e.g. segfault will not be recorded here:")
        for error in errors:
            print(error)

    if errors or invalid_results:
        sys.exit(1)


if __name__ == "__main__":
    main()
