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
)

from database_benchmarking_tools.experiment import ExperimentDB
from database_benchmarking_tools.utility import generate_db_path

import argparse
import importlib
import itertools


def query_identifier_to_name(identifier):
    # Convert "tpch_q6" -> "Q6"
    return identifier[5:].upper()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("location", help="TPC-H dataset location")
    arg_parser.add_argument("solution", help="Reference results location with pattern")
    args = arg_parser.parse_args()

    num_row_groups = 8
    load_all_data = 1
    storage = "memory"
    gqe_host = "localhost"
    query_source = "hand coded".lower()  # tool that generates the query plan
    query_source_path = query_source.replace(" ", "_")

    scale_factor = parse_scale_factor(args.location)

    edb_file = generate_db_path(f"gqe_{query_source_path}", "tpch", gqe_host)
    edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(
        GqeExperimentConnection
    )
    print(f"Writing SQLite file to {edb_file}")

    errors = []
    with edb_config as edb:
        edb_info = setup_db(edb, query_source)

        if load_all_data or (storage != "memory"):
            catalog = Catalog()
            catalog.register_tpch(args.location, storage, num_row_groups)

        for query_idx in [1, 2, 4, 6, 7, 11, 12, 15, 17, 18, 19, 20, 21]:
            if not load_all_data and (storage == "memory"):
                catalog = Catalog()
                catalog.register_tpch(args.location, storage, num_row_groups, query_idx)

            query_identifier = "tpch_q" + str(query_idx)
            module = importlib.import_module(query_identifier)
            root_relation = getattr(module, query_identifier)().root_relation()
            reference_file = args.solution.replace("%d", f"q{query_idx}")

            query = QueryInfo(
                query_identifier_to_name(query_identifier),
                root_relation,
                reference_file,
            )

            parameters = []
            for (
                num_partitions,
                read_use_zero_copy,
                max_num_workers,
            ) in itertools.product([1, 2, 4, 8], [False, True], [1]):
                if read_use_zero_copy and (num_partitions != num_row_groups):
                    continue

                parameters.append(
                    Parameter(num_partitions, read_use_zero_copy, max_num_workers)
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
