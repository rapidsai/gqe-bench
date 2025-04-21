#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    QueryInfo,
    Parameter,
    EdbInfo,
    setup_db,
    parse_scale_factor,
    set_eager_module_loading,
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
    args = arg_parser.parse_args()

    num_row_groups = 8
    load_all_data = 1
    storage = "memory"
    gqe_host = "localhost"
    query_source = args.query_source.lower()
    query_source_path = query_source.replace(" ", "_")
    use_opt_char_type = True

    # You can set it to int32 or int64, for SF1k int64 is required.
    identifier_type = lib.TypeId.int32
    scale_factor = parse_scale_factor(args.dataset)

    set_eager_module_loading()

    edb_file = args.output if args.output else generate_db_path(f"gqe_{query_source_path}", "tpch", gqe_host)
    edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(
        GqeExperimentConnection
    )
    print(f"Writing SQLite file to {edb_file}")

    errors = []
    with edb_config as edb:
        edb_info = setup_db(edb, query_source)

        if load_all_data or (storage != "memory"):
            catalog = Catalog()
            catalog.register_tpch(args.dataset, storage, num_row_groups, 0, identifier_type, use_opt_char_type)

        for query_idx in range(1, 23):

            if query_idx == 15 or query_idx == 18 or query_idx == 21 or query_idx == 1:
                continue

            if not load_all_data and (storage == "memory"):
                catalog = Catalog()
                catalog.register_tpch(args.dataset, storage, num_row_groups, query_idx, identifier_type, use_opt_char_type)

            reference_file = args.solution.replace("%d", f"q{query_idx}")

            substrait_file = os.path.join(args.plan, f"df_q{query_idx}.bin")
            root_relation = catalog.load_substrait(substrait_file)

            query = QueryInfo(f"Q{query_idx}", root_relation, reference_file)

            parameters = []
            for (
                num_partitions,
                read_use_zero_copy,
                max_num_workers,
                join_use_unique_keys,
            ) in itertools.product([1, 2, 4, 8], [False, True], [1], [True]):
                if read_use_zero_copy and (num_partitions != num_row_groups):
                    continue

                parameters.append(
                    Parameter(num_partitions, read_use_zero_copy, max_num_workers, join_use_unique_keys)
                )

            run_tpc(catalog, query, scale_factor, parameters, edb, edb_info, errors)

    print(f"Finished SQLite file at {edb_file}")

    if errors:
        print(
            "The following configurations run successfully but produce incorrect results"
        )
        print(errors)


if __name__ == "__main__":
    main()
