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
from gqe.benchmark.run import run_tpc, QueryInfo, Parameter
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
    arg_parser.add_argument("solution", help="Reference results location")
    args = arg_parser.parse_args()

    num_row_groups = 8
    perf_db_file = "gqe_tpch.db3"

    suffix = 1
    while os.path.isfile(perf_db_file):
        perf_db_file = f"gqe_tpch_{suffix}.db3"
        suffix += 1

    catalog = Catalog()
    catalog.register_tpch(args.location, "memory", num_row_groups)

    query_identifiers = ["tpch_q1", "tpch_q6", "tpch_q15", "tpch_q17", "tpch_q18", "tpch_q21"]
    queries = []
    for query_identifier in query_identifiers:
        module = importlib.import_module(query_identifier)
        root_relation = getattr(module, query_identifier)().root_relation()
        reference_file = os.path.join(args.solution, f"q{query_identifier[6:]}.parquet")

        queries.append(
            QueryInfo(query_identifier_to_name(query_identifier), root_relation, reference_file))

    parameters = []
    for num_partitions, read_use_zero_copy, max_num_workers in itertools.product(
            [1, 2, 4, 8], [False, True], [1]):
        if read_use_zero_copy and (num_partitions != num_row_groups):
            continue

        parameters.append(Parameter(num_partitions, read_use_zero_copy, max_num_workers))

    run_tpc(perf_db_file, catalog, queries, parameters)


if __name__ == "__main__":
    main()
