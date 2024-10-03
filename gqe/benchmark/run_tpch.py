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
from gqe.benchmark.run import run_tpc, QueryInfo, Parameter, connect_db
import argparse
import importlib
import os
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

    perf_db_file = "gqe_tpch.db3"
    suffix = 1
    while os.path.isfile(perf_db_file):
        perf_db_file = f"gqe_tpch_{suffix}.db3"
        suffix += 1

    out_cursor, out_conn, hw_info_id = connect_db(perf_db_file)
    errors = []
    
    if load_all_data or (storage != "memory"):
        catalog = Catalog()
        catalog.register_tpch(args.location, storage, num_row_groups)

    for query_idx in [1, 2, 6, 11, 12, 15, 17, 18, 20, 21]:
        if not load_all_data and (storage == "memory"):
            catalog = Catalog()
            catalog.register_tpch(args.location, storage, num_row_groups, query_idx)

        query_identifier = "tpch_q" + str(query_idx)
        module = importlib.import_module(query_identifier)
        root_relation = getattr(module, query_identifier)().root_relation()
        reference_file = args.solution.replace("%d", f"q{query_idx}")

        query = QueryInfo(
            query_identifier_to_name(query_identifier), root_relation, reference_file
        )

        parameters = []
        for num_partitions, read_use_zero_copy, max_num_workers in itertools.product(
            [1, 2, 4, 8], [False, True], [1]
        ):
            if read_use_zero_copy and (num_partitions != num_row_groups):
                continue

            parameters.append(
                Parameter(num_partitions, read_use_zero_copy, max_num_workers)
            )

        run_tpc(catalog, query, parameters, out_cursor, hw_info_id, errors)

    out_conn.commit()
    print(f"Finished SQLite file at {perf_db_file}")

    print("The following configurations run successfully but produce incorrect results")
    print(errors)


if __name__ == "__main__":
    main()
