#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import functools
import itertools
import json

# We rename this because of namespace clash with multiprocessing in GQE
import multiprocessing as subprocessing
import os
import signal
import sys
import time

from database_benchmarking_tools.experiment import ExperimentDB
from database_benchmarking_tools.utility import generate_db_path

from gqe import lib
from gqe.benchmark.gqe_experiment import GqeExperimentConnection
from gqe.benchmark.run import (
    CatalogContext,
    DataInfo,
    QueryExecutionContext,
    QueryInfoContext,
    boost_shared_memory_pool_size,
    is_valid_identifier_type,
    parse_bool,
    parse_identifier_type,
    parse_scale_factor,
    print_mp,
    run_tpc,
    set_eager_module_loading,
    setup_db,
)
from gqe.param_sweep_config import (
    BENCHMARK_CONFIG_DEFAULTS,
    QUERY_CONFIG_DEFAULTS,
    check_cli_overrides,
    config_to_args,
    get_query_execution_params,
    get_validation_dir,
    load_json_config,
)


def get_queries(query_source: str, queries: list[str] = None):
    handcoded_queries = [
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

    # 1, 15, 18, 21 removed as per previous run-script
    # Q11 FIXME: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/141
    # Q4 removed as the substrait plan fails for SF1k
    substrait_queries = [
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

    if queries:
        handcoded_queries = sorted(set(handcoded_queries) & set(queries))
        substrait_queries = sorted(set(substrait_queries) & set(queries))

    if query_source == "handcoded":
        if len(handcoded_queries) == 0:
            raise ValueError(f"No handcoded queries found for the given queries")
        substrait_queries = []
    elif query_source == "substrait":
        if len(substrait_queries) == 0:
            raise ValueError(f"No substrait queries found for the given queries")
        handcoded_queries = []
    elif query_source == "both":
        pass
    else:
        raise ValueError(f"Invalid query source: {query_source}")

    return handcoded_queries, substrait_queries


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
    arg_parser.add_argument("dataset", nargs="?", help="TPC-H dataset location")
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
        choices=["handcoded", "substrait", "both"],
        default=BENCHMARK_CONFIG_DEFAULTS["query_source"],
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
            "parquet_file",
        ],
        nargs="+",
        default=BENCHMARK_CONFIG_DEFAULTS["storage_kind"],
    )
    metrics_group.add_argument(
        "--multiprocess", "-m", help="Run in multiprocess mode", action="store_true"
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
        help="Timeout in (s) for queries in subprocesses. Used to kill suspected hanging jobs after timeout exceeded. Ignored if -sb is not set. Ignored if -m is set. Default: 30m",
        type=int,
        default=BENCHMARK_CONFIG_DEFAULTS["query_timeout"],
    )
    arg_parser.add_argument(
        "--data-timeout",
        "-dt",
        help="Timeout in (s) for data load subprocesses. Used to kill suspected hanging jobs after timeout exceeded. Ignored if -sb is not set. Ignored if -m is set. Default: 3hr",
        type=int,
        default=BENCHMARK_CONFIG_DEFAULTS["data_timeout"],
    )
    arg_parser.add_argument(
        "--boost_pool_size", help="Boost pool size in GBs", type=int, default=None
    )
    arg_parser.add_argument(
        "--sandboxing",
        "-sb",
        help="Run with sandboxing. Ignored if -m is set.",
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
        help="Use CPU compression for in-memory table compression",
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

    if args.query_timeout < 0 or args.data_timeout < 0:
        print(f"Timeouts cannot be negative: {args.query_timeout}, {args.data_timeout}")
        print("Timeout must be a positive integer or 0")
        print("Exiting with error")
        sys.exit(1)

    if args.sandboxing and args.multiprocess:
        print(
            f"Multi-process sandboxing enabled by -sb is not compatible with multi-gpu set by -m at this time. Ignoring sandboxing."
        )
    validate_dir = get_validation_dir(args.validate_dir)

    gqe_host = "localhost"
    scale_factor = parse_scale_factor(args.dataset)
    load_all_data = (
        args.load_all_data if args.load_all_data is not None else (1 if scale_factor <= 200 else 0)
    )

    multiprocess_runtime_context = None
    if args.multiprocess:
        if args.storage_kind != ["parquet_file"] and args.storage_kind != ["boost_shared_memory"]:
            raise ValueError(
                "Multiprocess mode is only supported with parquet_file storage kind or boost_shared_memory storage kind"
            )

        lib.mpi_init()

        if args.storage_kind == ["boost_shared_memory"]:
            pool_size = boost_shared_memory_pool_size(
                args.boost_pool_size, scale_factor, load_all_data
            )
            print(f"Initializing CPU shared memory with pool size {pool_size}")
            lib.initialize_shared_memory(pool_size)

        multiprocess_runtime_context = lib.MultiProcessRuntimeContext(
            lib.scheduler_type.ROUND_ROBIN, args.storage_kind[0]
        )

    if not args.identifier_type:
        identifier_type = [parse_identifier_type(args.dataset)]
    else:
        # You can set it to int32 or int64, for SF1k int64 is required.
        str_to_type = {"int32": lib.TypeId.int32, "int64": lib.TypeId.int64}
        identifier_type = [str_to_type[t] for t in args.identifier_type]

    set_eager_module_loading()

    edb_file = None
    edb_config = None
    edb_info = None
    is_root_rank = (not args.multiprocess) or (lib.mpi_rank() == 0)
    if is_root_rank:
        edb_file = args.output if args.output else generate_db_path(f"gqe", "tpch", gqe_host)

        edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(GqeExperimentConnection)
        edb_config.create_experiment_db()
        edb_info = None
        with edb_config as edb:
            edb_info = setup_db(edb)
        print(f"Writing SQLite file to {edb_file}")

    handcoded_queries, substrait_queries = get_queries(args.query_source, args.queries)

    def run_sweep(edb_file, edb_info, args):
        nonlocal identifier_type
        errors = []
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
            match is_valid_identifier_type(identifier_type, "tpch", scale_factor):
                case True:
                    pass
                case False:
                    continue
                case None:
                    raise ValueError(
                        f"Unknown if identifier type { identifier_type } is valid for the given dataset"
                    )

            data_info = DataInfo(
                storage_device_kind=storage_kind,
                format="internal",
                location=None,  # FIXME: set location as NUMA node, iff set in GQE
                not_null=False,
                identifier_type=str(identifier_type),
                char_type="char" if use_opt_type_for_single_char_col else "text",
                decimal_type="float",
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
            ]:
                for query_str in queries:
                    query_idx = int(query_str.split("_")[0])
                    reference_file = args.solution.replace("%d", f"q{query_idx}")
                    physical_plan_folder = None
                    if query_source == "handcoded":
                        substrait_file = None
                    elif query_source == "substrait":
                        substrait_file = os.path.join(args.plan, f"df_q{query_idx}.bin")
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
                                is_root_rank and not args.quiet,
                            )
                            continue

                        # Zero copy can only be used for in-memory tables.
                        if read_use_zero_copy and storage_kind == "parquet_file":
                            print_mp(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, storage_kind: {storage_kind} because zero copy is not supported for parquet files",
                                is_root_rank,
                            )
                            continue

                        # Skip zero copy for compression, as compression takes precedence over zero copy in GQE read task.
                        #
                        # Note: Revisit if GQE behavior changes in future.
                        if read_use_zero_copy and compression_format != "none":
                            print_mp(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, compression_format: {compression_format} because zero copy is not supported with compression",
                                is_root_rank and not args.quiet,
                            )
                            continue

                        if num_workers > num_partitions:
                            print_mp(
                                f"Skipping num_workers: {num_workers}, num_partitions: {num_partitions} because num_workers greater than num_partitions",
                                is_root_rank and not args.quiet,
                            )
                            continue

                        if compression_format != "none" and use_overlap_mtx:
                            print_mp(
                                f"Skipping compression_format: {compression_format}, use_overlap_mtx: {use_overlap_mtx} because its not optimal to use overlap matrix with compression",
                                is_root_rank and not args.quiet,
                            )
                            continue

                        # We want to always use hash map caching except when perfect hashing is used
                        # Because perfect join doesn't support hash map cache, see: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/161
                        if join_use_perfect_hash == join_use_hash_map_cache:
                            print_mp(
                                f"Skipping join_use_perfect_hash: {join_use_perfect_hash}, join_use_hash_map_cache: {join_use_hash_map_cache} because hash map caching should always be used except with perfect hashing",
                                is_root_rank and not args.quiet,
                            )
                            continue

                        # Perfect hash join is disabled for substrait plans, see: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/161
                        if query_source == "substrait" and (
                            join_use_perfect_hash or aggregation_use_perfect_hash
                        ):
                            print_mp(
                                f"Skipping join_use_perfect_hash: {join_use_perfect_hash}, aggregation_use_perfect_hash: {aggregation_use_perfect_hash}, query_source: {query_source}. Because, perfect hash is only manually enabled in physical plans",
                                is_root_rank and not args.quiet,
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
                        ):
                            print_mp(
                                f"Skipping num_partitions: {num_partitions}, num_row_groups: {num_row_groups}, scale_factor: {scale_factor} because configuration is likely to encounter cuDF concatenate error",
                                is_root_rank and not args.quiet,
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
            parameters.sort(key=lambda p: p.query_info_ctx.query_idx)
            # We use the non-sandboxing path if sandboxing is not set, or if multiprocessing/multigpu is set.
            if not args.sandboxing or args.multiprocess:
                print_mp(
                    "Running experiments without subprocess sandbox",
                    is_root_rank and not args.quiet,
                )
                run_tpc(
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
                    is_root_rank,
                    args.multiprocess,
                    multiprocess_runtime_context,
                    args.validate_results,
                    validate_dir,
                    args.quiet,
                )

            # Subprocess Case
            # Ideal case: spawn process, run to completion, join.
            # If subprocess fails, we iterate and launch the process again w/ the remaining parameters.
            # Subprocess MUST remove tasks from list for this to function correctly.
            else:
                subproc_ctx = subprocessing.get_context("spawn")
                with subproc_ctx.Manager() as manager:
                    parameter_queue = manager.list(parameters)
                    # explicitly replace errors list with list proxy
                    errors = manager.list()
                    invalid_results = manager.list()
                    print_mp("Running experiments with subprocess sandbox", is_root_rank)
                    while parameter_queue:
                        parent_pipe, child_pipe = subproc_ctx.Pipe()
                        subproc = subproc_ctx.Process(
                            target=run_tpc,
                            args=(
                                cat_ctx,
                                data_info,
                                scale_factor,
                                parameter_queue,
                                edb_file,
                                edb_info,
                                args.metrics,
                                errors,
                                invalid_results,
                                args.repeat,
                                is_root_rank,
                                args.multiprocess,
                                multiprocess_runtime_context,
                                args.validate_results,
                                validate_dir,
                                args.quiet,
                                child_pipe,
                            ),
                        )
                        subprocess_run(
                            subproc,
                            parameter_queue,
                            load_all_data,
                            is_root_rank,
                            parent_pipe,
                            args,
                        )
                    # convert to regular list before manager destructs
                    all_invalid_results += list(invalid_results)
                    all_errors += list(errors)
        # Return should match outer loop indentation
        return all_errors, all_invalid_results

    errors = []
    errors, invalid_results = run_sweep(edb_file, edb_info, args)
    print_mp(f"Finished SQLite file at {edb_file}", is_root_rank)

    if invalid_results:
        print_mp(
            f"The following {len(invalid_results)} configurations run successfully but produce incorrect results",
            is_root_rank,
        )
        for result in invalid_results:
            print_mp(result, is_root_rank)

    if errors:
        print_mp(f"The following {len(errors)} configurations produced errors", is_root_rank)
        print_mp("note: hard crashes e.g. segfault will not be recorded here:", is_root_rank)
        for error in errors:
            print_mp(error, is_root_rank)

    if args.multiprocess:
        multiprocess_runtime_context.finalize()
        if args.storage_kind == "boost_shared_memory" and lib.mpi_rank() == 0:
            lib.finalize_shared_memory()
        lib.mpi_finalize()

    if errors or invalid_results:
        sys.exit(1)


def subprocess_kill(subproc, message, is_root_rank):
    print_mp(message, is_root_rank)
    subproc.terminate()
    time.sleep(1)
    # If process is still alive, try to kill more forcefully.
    if subproc.is_alive():
        print_mp("Process still alive after terminate, trying sigkill", is_root_rank)
        subproc.kill()
        time.sleep(1)
        # if subprocess is still alive after here something is really bad
        # TODO investigate ways to mitigate if it becomes an issue


def do_poll(subproc, pipe, timeout):
    start = time.monotonic()
    current = time.monotonic()
    poll_time = 1
    result = False
    # Iterate while timeout isn't reached, experiments are still in queue, and the subprocess is alive.
    while current - start < timeout and subproc.is_alive() and not result:
        result = pipe.poll(poll_time)
        current = time.monotonic()
    elapsed = current - start
    return result, elapsed


def subprocess_run(subproc, parameter_queue, load_all_data, is_root_rank, pipe, args):
    query_timeout = args.query_timeout
    data_timeout = args.data_timeout
    print(f"Starting parameter set execution with {len(parameter_queue)} sets remaining")
    print(f"Using {query_timeout}s query timeout and {data_timeout}s data load timeout")
    subproc.start()

    killed = False
    iters = 0
    # Subprocess must pop items from queue for this to work properly
    while parameter_queue and not killed:
        print_mp(
            f"Parameter sets remaining: {len(parameter_queue)}",
            is_root_rank,
        )
        iters += 1
        # Stage 1: Wait on data load. Two Cases
        # load_all_data = True; the first iter, this determines success. After, we always expect success (no data to load).
        # load_all_data = False; we don't always load data, so we get true if it succeeded or didn't load, otherwise false.
        data_avail, elapsed = do_poll(subproc, pipe, data_timeout)
        print_mp(f"Data load stage ended after {elapsed:.2f}s", is_root_rank)
        if not data_avail:
            if elapsed >= data_timeout:
                subprocess_kill(
                    subproc,
                    f"Timeout triggered - data load did not complete within {data_timeout} seconds",
                    is_root_rank,
                )
                killed = True
                if load_all_data:
                    print_mp(
                        "Because this is load_all_data, dropping dependent experiments",
                        is_root_rank,
                    )
                    parameter_queue[:] = []
            # always break if we get here to move on to new process
            # the other option is we're not alive, so we break anyway
            break
        # if we get a false on receive, it means data loading failed for some reason
        if not pipe.recv():
            # Since load_all_data failed, we need to discard the experiments or we will not make forward progress.
            if load_all_data:
                parameter_queue[:] = []
                break
            # if loading only query data, safe to proceed to next query and try again
            else:
                continue

        # Stage 2: wait on context build
        data_avail, elapsed = do_poll(subproc, pipe, query_timeout)
        print_mp(f"Context creation ended after {elapsed:.2f}s", is_root_rank)
        if not data_avail:
            if elapsed >= query_timeout:
                killed = True
                subprocess_kill(
                    subproc,
                    f"Timeout triggered - query did not complete within {query_timeout} seconds",
                    is_root_rank,
                )
            # always break if we get here to move on to new process
            break
        # If pipe sends false, it means we failed to load context and move on to next parameter set.
        if not pipe.recv():
            continue

        # Stage 3: Wait on query completion
        for i in range(args.repeat):
            # Wait on query process
            data_avail, elapsed = do_poll(subproc, pipe, query_timeout)
            print_mp(
                f"Query execution/validation iteration {i} ended after {elapsed:.2f}s",
                is_root_rank,
            )
            if not data_avail:
                if elapsed >= query_timeout:
                    killed = True
                    subprocess_kill(
                        subproc,
                        f"Timeout triggered - query did not complete within {query_timeout} seconds",
                        is_root_rank,
                    )
                # always break if we get here to move on to new process
                break
            # If pipe sends false, it generally means a query error or query validation failed.
            if not pipe.recv():
                # Both multiprocessing and single gpu break out of query or query validation if error.
                # Note that this only breaks the inner loop - all other cases continue
                break
    subproc.join()

    # Breaking above will always take you here, which evaluates the subprocess execution status.
    if subproc.exitcode != 0:
        print_mp(
            f"Subprocess exited with non-zero return code with {len(parameter_queue)} tasks remaining.",
            is_root_rank,
        )
        print_mp(f"Subprocess Exit Code {subproc.exitcode}", is_root_rank)
        # We won't print the "may have" part if we deliberately killed the subprocess ourselves.
        if subproc.exitcode < 0 and not killed:
            print_mp(
                f"Subprocess may have been killed by {signal.Signals(-subproc.exitcode).name}",
                is_root_rank,
            )


if __name__ == "__main__":
    main()
