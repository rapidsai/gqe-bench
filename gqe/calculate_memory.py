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
from typing import Dict

import gqe.lib
from gqe import table_definition


def get_row_counts(scale_factor: float) -> Dict[str, int]:
    return {
        "lineitem": round(scale_factor * 6_000_000),
        "orders": round(scale_factor * 1_500_000),
        "part": round(scale_factor * 200_000),
        "partsupp": round(scale_factor * 800_000),
        "customer": round(scale_factor * 150_000),
        "supplier": round(scale_factor * 10_000),
        "nation": 25,
        "region": 5,
    }


def get_type_sizes() -> Dict[gqe.lib.TypeId, int]:
    return {
        gqe.lib.TypeId.int32: 4,
        gqe.lib.TypeId.int64: 8,
        gqe.lib.TypeId.float32: 4,
        gqe.lib.TypeId.float64: 8,
        gqe.lib.TypeId.timestamp_days: 4,  # date type
        gqe.lib.TypeId.string: 25,
        gqe.lib.TypeId.int8: 1,  # char type
    }


def calculate_memory_requirements(
    definitions: dict[str, list[gqe.lib.ColumnTraits]], scale_factor: float
) -> (int, int):
    type_sizes = get_type_sizes()
    row_counts = get_row_counts(scale_factor)

    total_memory = 0
    table_to_mem_usage = {}

    for table_name, columns in definitions.items():
        row_size = sum(type_sizes[column_traits.data_type.type_id()] for column_traits in columns)
        table_to_mem_usage[table_name] = row_size * row_counts[table_name]
    total_memory = sum(table_to_mem_usage.values())

    return total_memory, table_to_mem_usage


def estimate_memory_for_all_queries(
    scale_factor: float, identifier_type: gqe.lib.TypeId, use_opt_char_type: bool
):
    table_defs = table_definition.TPCHTableDefinitions(identifier_type, use_opt_char_type)

    for query_idx in range(23):  # 0-22 inclusive
        definitions = table_defs.query_table_definitions(query_idx)
        memory_needed_for_query, memory_needed_by_table = calculate_memory_requirements(
            definitions, scale_factor
        )

        if query_idx == 0:
            print("  Total", end="")
        else:
            print(f"Query {query_idx}", end="")

        print(f" memory needed: {memory_needed_for_query / (1024 * 1024 * 1024):.2f} GiB")
        for table_name, mem_needed in memory_needed_by_table.items():
            print(
                f"  Table: {table_name}, memory needed: {mem_needed / (1024 * 1024 * 1024):.3f} GiB"
            )


def main():
    arg_parser = argparse.ArgumentParser(
        description="A script to calculate the approximate memory requirement for TPC-H queries."
    )

    arg_parser.add_argument(
        "--scale-factor",
        "-s",
        required=True,
        type=float,
        help="Scale factor of the input data.",
    )
    arg_parser.add_argument(
        "--identifier-type",
        "-i",
        help="Identifier type used in the dataset.",
        choices=["int32", "int64"],
        type=str,
        required=True,
    )
    arg_parser.add_argument(
        "--use-opt-char-type",
        "-u",
        help="Use int8 (1) or string (0) to represent char type. Default: 1",
        type=int,
        choices=[0, 1],
        default=1,
    )

    args = arg_parser.parse_args()

    str_to_type = {"int32": gqe.lib.TypeId.int32, "int64": gqe.lib.TypeId.int64}
    identifier_type = str_to_type[args.identifier_type]

    estimate_memory_for_all_queries(args.scale_factor, identifier_type, args.use_opt_char_type)


if __name__ == "__main__":
    main()
