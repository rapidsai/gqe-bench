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
    parse_scale_factor,
    parse_identifier_type,
    set_eager_module_loading,
    is_valid_identifier_type,
)
from gqe import lib


from database_benchmarking_tools.experiment import ExperimentDB
from database_benchmarking_tools.utility import generate_db_path

import argparse
import os
import itertools


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", help="TPC-H dataset location")
    arg_parser.add_argument("plan", help="Substrait query plan location")
    arg_parser.add_argument("solution", help="Reference results location with pattern")
    arg_parser.add_argument("--output", "-o", help="Output file path")
    arg_parser.add_argument(
        "--query-source",
        help="Tool that generated the query plan",
        default="datafusion",
    )
    arg_parser.add_argument(
        "--queries",
        "-q",
        help="Which queries to run",
        nargs="+",
        action="extend",
        type=int,
    )
    arg_parser.add_argument(
        "--identifier-type",
        "-i",
        help="Identifier type used in the dataset",
        choices=["int32", "int64"],
        nargs="+",
        action="extend",
        type=str,
    )
    arg_parser.add_argument(
        "--compression_format",
        "-c",
        help="Compression format to use",
        choices=[
            "none", "ans", "lz4", "snappy", "gdeflate", "deflate", 
            "cascaded", "zstd", "gzip", "bitcomp", 
            "best_compression_ratio", "best_decompression_speed"
        ],
        nargs="+",
        action="extend",
        type=str,
        default=["none"],
    )
    args = arg_parser.parse_args()

    load_all_data = 1
    gqe_host = "localhost"
    query_source = args.query_source.lower()
    query_source_path = query_source.replace(" ", "_")
    scale_factor = parse_scale_factor(args.dataset)

    # You can set it to int32 or int64, for SF1k int64 is required.
    str_to_type = {"int32": lib.TypeId.int32, "int64": lib.TypeId.int64}
    if not args.identifier_type:
        identifier_type = [parse_identifier_type(args.dataset)]
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
        ) in itertools.product(
            num_row_group_list,
            [True],
            args.compression_format,
            ["char"],
            [2**16],
            [lib.TypeId.int32] if scale_factor < 357 else [lib.TypeId.int64],
            ["pinned_memory"],
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
                    )
                except Exception as e:
                    print(f"Error registering table: {e}")
                    continue

            queries = (
                args.queries
                if args.queries
                else range(1, 23)
            )
            for query_idx in queries:

                if (
                    query_idx == 15
                    or query_idx == 18
                    or query_idx == 21
                    or query_idx == 1
                ):
                    continue

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
                        )
                    except Exception as e:
                        print(
                            f"Error registering in memory table for query {query_idx}: {e}"
                        )
                        continue

                reference_file = args.solution.replace("%d", f"q{query_idx}")

                substrait_file = os.path.join(args.plan, f"df_q{query_idx}.bin")
                root_relation = catalog.load_substrait(substrait_file)

                query = QueryInfo(f"Q{query_idx}", root_relation, reference_file)

                parameters = []
                for (
                    num_workers,
                    num_partitions,
                    use_overlap_mtx,
                    join_use_hash_map_cache,
                    read_use_zero_copy,
                    join_use_unique_keys,
                ) in itertools.product(
                    # TODO Change num_workers to [1, 2, 4] when https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/153 is fixed
                    [1], [1, 2, 4, 8], [True], [False], [False, True], [True]
                ):
                    # Skip zero copy for partition-row-group combinations where zero copy is not supported.
                    if read_use_zero_copy and (num_partitions != num_row_groups):
                        continue

                    # Skip zero copy for compression, as compression takes precedence over zero copy in GQE read task.
                    #
                    # Note: Revisit if GQE behavior changes in future.
                    if read_use_zero_copy and compression_format != "none":
                        continue

                    parameters.append(
                        Parameter(
                            num_workers,
                            num_partitions,
                            use_overlap_mtx,
                            join_use_hash_map_cache,
                            read_use_zero_copy,
                            join_use_unique_keys,
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
