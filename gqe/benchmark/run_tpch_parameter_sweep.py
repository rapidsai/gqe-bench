#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import Catalog
from gqe.benchmark.gqe_experiment import GqeExperimentConnection
from gqe.benchmark.run import (
    run_tpc,
    run_tpc_multiprocess,
    DataInfo,
    QueryInfo,
    Parameter,
    setup_db,
    parse_scale_factor,
    set_eager_module_loading,
    parse_bool,
    parse_identifier_type,
    is_valid_identifier_type,
    fix_partial_filter_column_references,
    get_query_validator,
    boost_shared_memory_pool_size,
)
from gqe import lib


from database_benchmarking_tools.experiment import ExperimentDB
from database_benchmarking_tools.utility import generate_db_path

import argparse
import importlib
import itertools
import os


def get_queries(query_source: str, queries: list[str] = None):
    handcoded_queries = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "9",
        "10",
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
        "19",
        "20",
        "21",
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


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", help="TPC-H dataset location")
    arg_parser.add_argument("plan", help="Substrait query plan location")
    arg_parser.add_argument("solution", help="Reference results location with pattern")
    arg_parser.add_argument("--output", "-o", help="Output file path")
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
    arg_parser.add_argument(
        "--row-groups",
        "-r",
        help="Number of row groups to use",
        nargs="+",
        action="extend",
        type=int,
    )
    arg_parser.add_argument(
        "--partitions",
        "-p",
        help="Number of partitions to use",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )
    arg_parser.add_argument(
        "--workers",
        "-w",
        help="Number of workers to use",
        nargs="+",
        type=int,
        default=[1],
    )
    arg_parser.add_argument(
        "--query-source",
        help="Query source",
        choices=["handcoded", "substrait", "both"],
        default="both",
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
        default=["none"],
    )
    arg_parser.add_argument(
        "--load-all-data",
        "-l",
        help="Whether to load all data at once (1) or per query (0). If not specified, defaults to 1 for scale_factor <= 200, 0 otherwise",
        type=int,
        choices=[0, 1],
    )
    arg_parser.add_argument(
        "--partition-pruning",
        "-spp",
        help="Enable partition pruning optimization",
        type=parse_bool,
        nargs="*",
        default=[False],
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
        default=["numa_pinned_memory"],
    )
    arg_parser.add_argument(
        "--multiprocess", "-m", help="Run in multiprocess mode", action="store_true"
    )
    arg_parser.add_argument("--boost_pool_size", help="Boost pool size in GBs", type=int, default=None)

    args = arg_parser.parse_args()

    gqe_host = "localhost"
    scale_factor = parse_scale_factor(args.dataset)
    load_all_data = (
        args.load_all_data
        if args.load_all_data is not None
        else (1 if scale_factor <= 200 else 0)
    )

    if args.multiprocess:
        if args.storage_kind != ["parquet_file"] and args.storage_kind != ["boost_shared_memory"]:
            raise ValueError(
                "Multiprocess mode is only supported with parquet_file storage kind or boost_shared_memory storage kind"
            )
        
        lib.mpi_init()
        
        if args.storage_kind == ["boost_shared_memory"]:
            pool_size = boost_shared_memory_pool_size(args.boost_pool_size, scale_factor, load_all_data)
            print(f"Initializing CPU shared memory with pool size {pool_size}")
            lib.initialize_shared_memory(pool_size)
        
        multiprocess_runtime_context = lib.MultiProcessRuntimeContext(lib.scheduler_type.ROUND_ROBIN, args.storage_kind[0])
        
        



    if not args.identifier_type:
        identifier_type = [parse_identifier_type(args.dataset)]
    else:
        # You can set it to int32 or int64, for SF1k int64 is required.
        str_to_type = {"int32": lib.TypeId.int32, "int64": lib.TypeId.int64}
        identifier_type = [str_to_type[t] for t in args.identifier_type]

    set_eager_module_loading()

    edb_file = None
    edb_config = None
    is_root_rank = (not args.multiprocess) or (lib.mpi_rank() == 0)
    if is_root_rank:
        edb_file = (
            args.output if args.output else generate_db_path(f"gqe", "tpch", gqe_host)
        )

        edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(
            GqeExperimentConnection
        )

        print(f"Writing SQLite file to {edb_file}")

    handcoded_queries, substrait_queries = get_queries(args.query_source, args.queries)

    def run_sweep(edb, edb_info, args):
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

        for (
            num_row_groups,
            use_opt_type_for_single_char_col,
            compression_format,
            compression_data_type,
            compression_chunk_size,
            identifier_type,
            storage_kind,
            zone_map_partition_size,
        ) in itertools.product(
            num_row_group_list,
            [True],
            args.compression_format,
            ["char"],
            [2**16],
            identifier_type,
            args.storage_kind,
            [100000],
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
                compression_data_type=compression_data_type,
                compression_chunk_size=compression_chunk_size,
                zone_map_partition_size=zone_map_partition_size,
            )

            table_definitions = None
            if load_all_data or (storage_kind == "parquet_file"):
                catalog = Catalog()
                try:
                    table_definitions = catalog.register_tpch(
                        args.dataset,
                        storage_kind,
                        num_row_groups,
                        0,
                        identifier_type,
                        use_opt_type_for_single_char_col,
                        compression_format,
                        compression_data_type,
                        compression_chunk_size,
                        zone_map_partition_size,
                        multiprocess_runtime_context if args.multiprocess else None,
                    )
                except Exception as e:
                    print(f"Error registering table: {e}")
                    errors.append(f"Error registering table: {e}")
                    continue

            for query_source, queries in [
                ("handcoded", handcoded_queries),
                ("substrait", substrait_queries),
            ]:
                for query_str in queries:
                    query_idx = int(query_str.split("_")[0])
                    if not load_all_data and (storage_kind != "parquet_file"):
                        catalog = Catalog()
                        try:
                            table_definitions = catalog.register_tpch(
                                args.dataset,
                                storage_kind,
                                num_row_groups,
                                query_idx,
                                identifier_type,
                                use_opt_type_for_single_char_col,
                                compression_format,
                                compression_data_type,
                                compression_chunk_size,
                                zone_map_partition_size,
                                multiprocess_runtime_context if args.multiprocess else None
                            )
                        except Exception as e:
                            print(
                                f"Error registering in memory table for query {query_str}: {e}"
                            )
                            continue

                    reference_file = args.solution.replace("%d", f"q{query_idx}")

                    if query_source == "handcoded":
                        query_identifier = "tpch_q" + query_str
                        module = importlib.import_module(
                            f"gqe.benchmark.{query_identifier}"
                        )
                        query_object = getattr(module, query_identifier)(
                            scale_factor=scale_factor
                        )
                        root_relation = query_object.root_relation(table_definitions)
                        # Fix partial filter column references for handcoded queries
                        if not load_all_data and (storage_kind != "parquet_file"):
                            fix_partial_filter_column_references(
                                root_relation, query_idx
                            )
                        query = QueryInfo(
                            f"Q{query_str}", root_relation, reference_file
                        )
                        if validator := get_query_validator(query_object):
                            query.validator = validator

                    elif query_source == "substrait":
                        substrait_file = os.path.join(args.plan, f"df_q{query_idx}.bin")
                        root_relation = catalog.load_substrait(substrait_file, True, multiprocess_runtime_context if args.multiprocess else None)
                        query = QueryInfo(
                            f"Q{query_idx}", root_relation, reference_file
                        )

                    parameters = []
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
                        args.workers,
                        args.partitions,
                        [False, True],
                        [False, True],
                        [False, True],
                        [True],
                        [False, True],
                        [False, True],
                        args.partition_pruning,
                        [False, True],
                        [False, True],
                    ):
                        # Skip zero copy for partition-row-group combinations where zero copy is not supported.
                        if read_use_zero_copy and (num_partitions != num_row_groups):
                            print(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, num_partitions: {num_partitions}, num_row_groups: {num_row_groups} because zero copy requires row groups to be equal to partitions"
                            )
                            continue

                        # Zero copy can only be used for in-memory tables.
                        if read_use_zero_copy and storage_kind == "parquet_file":
                            print(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, storage_kind: {storage_kind} because zero copy is not supported for parquet files"
                            )
                            continue

                        # Skip zero copy for compression, as compression takes precedence over zero copy in GQE read task.
                        #
                        # Note: Revisit if GQE behavior changes in future.
                        if read_use_zero_copy and compression_format != "none":
                            print(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, compression_format: {compression_format} because zero copy is not supported with compression"
                            )
                            continue

                        if num_workers > num_partitions:
                            print(
                                f"Skipping num_workers: {num_workers}, num_partitions: {num_partitions} because num_workers greater than num_partitions"
                            )
                            continue

                        if compression_format != "none" and use_overlap_mtx:
                            print(
                                f"Skipping compression_format: {compression_format}, use_overlap_mtx: {use_overlap_mtx} because its not optimal to use overlap matrix with compression"
                            )
                            continue

                        # We want to always use hash map caching except when perfect hashing is used
                        # Because perfect join doesn't support hash map cache, see: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/161
                        if join_use_perfect_hash == join_use_hash_map_cache:
                            print(
                                f"Skipping join_use_perfect_hash: {join_use_perfect_hash}, join_use_hash_map_cache: {join_use_hash_map_cache} because hash map caching should always be used except with perfect hashing"
                            )
                            continue

                        # Perfect hash join is disabled for substrait plans, see: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/161
                        if query_source == "substrait" and (
                            join_use_perfect_hash or aggregation_use_perfect_hash
                        ):
                            print(
                                f"Skipping join_use_perfect_hash: {join_use_perfect_hash}, aggregation_use_perfect_hash: {aggregation_use_perfect_hash}, query_source: {query_source}. Because, perfect hash is only manually enabled in physical plans"
                            )
                            continue

                        parameters.append(
                            Parameter(
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
                            )
                        )

                    if args.multiprocess:
                        run_tpc_multiprocess(
                            multiprocess_runtime_context,
                            catalog,
                            data_info,
                            query,
                            scale_factor,
                            parameters,
                            edb,
                            edb_info,
                            errors,
                            query_source,
                            is_root_rank,
                        )
                    else:
                        run_tpc(
                            catalog,
                            data_info,
                            query,
                            scale_factor,
                            parameters,
                            edb,
                            edb_info,
                            errors,
                            query_source,
                        )
                    
        
        return errors

    errors = []
    if is_root_rank:
        with edb_config as edb:
            edb_info = setup_db(edb)
            errors = run_sweep(edb, edb_info, args)
        print(f"Finished SQLite file at {edb_file}")
    else:
        errors = run_sweep(None, None, args)

    if errors:
        print(
            "The following configurations run successfully but produce incorrect results"
        )
        print(errors)

    if args.multiprocess:
        multiprocess_runtime_context.finalize()
        if args.storage_kind == "boost_shared_memory" and lib.mpi_rank() == 0:
            lib.finalize_shared_memory()
        lib.mpi_finalize()

    if errors:
        exit(1)




if __name__ == "__main__":
    main()
