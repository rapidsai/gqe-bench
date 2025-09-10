#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    DataInfo,
    QueryInfo,
    Parameter,
    EdbInfo,
    setup_db,
    parse_bool,
    parse_scale_factor,
    parse_identifier_type,
    set_eager_module_loading,
    is_valid_identifier_type,
    fix_partial_filter_column_references,
)
from gqe import lib


from database_benchmarking_tools.experiment import ExperimentDB
from database_benchmarking_tools.utility import generate_db_path

import argparse
import importlib
import itertools


def query_identifier_to_name(identifier):
    # Convert "tpch_q6" -> "Q6"
    return identifier[5:].upper()


def get_query_validator(query_object):
    try:
        query_validator = getattr(query_object, "validate_query")
        # Check that the validator is a callable method.
        if not callable(query_validator):
            raise TypeError("Expected the query validator to be a callable method")
        else:
            return query_validator

    except AttributeError:
        # The query doesn't have a validator.
        return None


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", help="TPC-H dataset location")
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
        action="extend",
        type=str,
        default=None,
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
        "-p",
        help="Enable partition pruning optimization",
        type=parse_bool,
        nargs="*",
        default=[False],
    )
    args = arg_parser.parse_args()

    gqe_host = "localhost"
    query_source = "hand coded".lower()  # tool that generates the query plan
    query_source_path = query_source.replace(" ", "_")
    scale_factor = parse_scale_factor(args.dataset)
    load_all_data = (
        args.load_all_data
        if args.load_all_data is not None
        else (1 if scale_factor <= 200 else 0)
    )

    # Set default compression format if none provided
    if args.compression_format is None:
        args.compression_format = ["none"]

    # You can set it to int32 or int64, for SF1k int64 is required.
    str_to_type = {"int32": lib.TypeId.int32, "int64": lib.TypeId.int64}
    if not args.identifier_type:
        identifier_type = [lib.TypeId.int32] if scale_factor < 357 else [lib.TypeId.int64]
    else:
        identifier_type = [str_to_type[t] for t in args.identifier_type]

    set_eager_module_loading()

    edb_file = (
        args.output
        if args.output
        else generate_db_path(f"gqe_{query_source_path}", "tpch", gqe_host)
    )
    edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(
        GqeExperimentConnection
    )
    print(f"Writing SQLite file to {edb_file}")

    errors = []
    with edb_config as edb:
        edb_info = setup_db(edb, query_source)

        num_row_group_list = []
        if scale_factor < 500:
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
            ["numa_pinned_memory"],
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

            if load_all_data or (storage_kind == "parquet_file"):
                catalog = Catalog()
                try:
                    catalog.register_tpch(
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
                    )
                except Exception as e:
                    print(f"Error registering table: {e}")
                    continue
            
            complete_queries_as_int = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 15, 17, 18, 19, 20, 21, 22]
            complete_queries = [str(x) for x in complete_queries_as_int]
            customized_queries = ["22_opt"]
            complete_queries.extend(customized_queries)
            queries = (
                args.queries
                if args.queries
                else complete_queries
            )
            for query_ident in queries:
                query_identifier = "tpch_q" + str(query_ident)
                query_idx = int(query_ident.split("_")[0]) if "_" in query_ident else int(query_ident)
                module = importlib.import_module(query_identifier)
                # Set the scale factor; required by TPC-H Q11
                query_object = getattr(module, query_identifier)(
                    scale_factor=scale_factor
                )
                root_relation = query_object.root_relation()
                reference_file = args.solution.replace("%d", f"q{query_idx}")

                query = QueryInfo(
                    query_identifier_to_name(query_identifier),
                    root_relation,
                    reference_file,
                )

                if validator := get_query_validator(query_object):
                    query.validator = validator

                if not load_all_data and (storage_kind != "parquet_file"):
                    catalog = Catalog()
                    try:
                        catalog.register_tpch(
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
                        )
                        fix_partial_filter_column_references(root_relation, query_idx)
                    except Exception as e:
                        print(
                            f"Error registering in memory table for query {query_ident}: {e}"
                        )
                        continue

                parameters = []
                for (
                    num_workers,
                    num_partitions,
                    use_overlap_mtx,
                    join_use_hash_map_cache,
                    read_use_zero_copy,
                    join_use_unique_keys,
                    join_use_perfect_hash,
                    use_partition_pruning,
                    filter_use_like_shift_and,
                ) in itertools.product(
                    # TODO Change num_workers to [1, 2, 4] when https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/153 is fixed
                    [1],
                    [1, 2, 4, 8],
                    [True],
                    [True, False],
                    [False, True],
                    [True],
                    [True, False],
                    args.partition_pruning,
                    [True], # filter_use_like_shift_and
                ):
                    # Skip zero copy for partition-row-group combinations where zero copy is not supported.
                    if read_use_zero_copy and (num_partitions != num_row_groups):
                        continue

                    # Skip zero copy for compression, as compression takes precedence over zero copy in GQE read task.
                    #
                    # Note: Revisit if GQE behavior changes in future.
                    if read_use_zero_copy and compression_format != "none":
                        continue

                    # Perfect hash join does not currently support hash map cache
                    # We only run the configuration when one of the two is enabled
                    if join_use_perfect_hash == join_use_hash_map_cache:
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
                            use_partition_pruning,
                            filter_use_like_shift_and,
                        )
                    )
                run_tpc(
                    catalog,
                    data_info,
                    query,
                    scale_factor,
                    parameters,
                    edb,
                    edb_info,
                    errors,
                )

    print(f"Finished SQLite file at {edb_file}")

    if errors:
        print(
            "The following configurations run successfully but produce incorrect results"
        )
        print(errors)


if __name__ == "__main__":
    main()
