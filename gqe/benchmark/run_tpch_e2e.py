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
from gqe.benchmark.run import run_tpc, connect_db, QueryInfo, Parameter

import argparse
import os
import itertools


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", help="TPC-H dataset location")
    arg_parser.add_argument("plan", help="Substrait query plan location")
    arg_parser.add_argument("solution", help="Reference results location with pattern")
    args = arg_parser.parse_args()

    num_row_groups = 8
    load_all_data = 1
    storage = "memory"

    perf_db_file = "gqe_tpch_e2e.db3"

    suffix = 1
    while os.path.isfile(perf_db_file):
        perf_db_file = f"gqe_tpch_e2e_{suffix}.db3"
        suffix += 1

    out_cursor, out_conn, hw_info_id = connect_db(perf_db_file)
    errors = []

    if load_all_data or (storage != "memory"):
        catalog = Catalog()
        catalog.register_tpch(args.dataset, storage, num_row_groups)

    for query_idx in range(1, 23):

        if query_idx == 15 or query_idx == 18 or query_idx == 21 or query_idx == 1:
            continue

        if not load_all_data and (storage == "memory"):
            catalog = Catalog()
            catalog.register_tpch(args.dataset, storage, num_row_groups, query_idx)

        reference_file = args.solution.replace("%d", f"q{query_idx}")

        substrait_file = os.path.join(args.plan, f"df_q{query_idx}.bin")
        root_relation = catalog.load_substrait(substrait_file)

        query = QueryInfo(f"Q{query_idx}", root_relation, reference_file)

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
