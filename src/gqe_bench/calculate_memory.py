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

"""Approximate in-memory footprint of the TPC-H tables for a given configuration.

Row counts come from the TPC-H spec's per-table cardinalities scaled by the scale
factor; column widths come from ``TpchTableSchema``, so the estimate follows the
same identifier and decimal choices a run would make.
"""

import argparse

from gqe_bench.physical_plan.expression import DataType, DataTypeId
from gqe_bench.suites.tpch.table_schema import TpchTableSchema

_BYTES_PER_GIB = 1024 * 1024 * 1024

# Average width assumed for a variable-length string column.
_STRING_WIDTH = 25

_TYPE_SIZES: dict[DataTypeId, int] = {
    DataTypeId.INT8: 1,  # char type
    DataTypeId.INT32: 4,
    DataTypeId.INT64: 8,
    DataTypeId.FLOAT32: 4,
    DataTypeId.FLOAT64: 8,
    DataTypeId.TIMESTAMP_DAYS: 4,  # date type
    DataTypeId.STRING: _STRING_WIDTH,
    DataTypeId.DECIMAL32: 4,
    DataTypeId.DECIMAL64: 8,
    DataTypeId.DECIMAL128: 16,
}


def get_row_counts(scale_factor: float) -> dict[str, int]:
    """Return the TPC-H per-table row counts at ``scale_factor``."""
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


def _type_size(data_type: DataType) -> int:
    """Return the per-value byte width for a column type."""
    return _TYPE_SIZES[data_type.type_id]


def calculate_memory_requirements(
    schema: TpchTableSchema, scale_factor: float
) -> tuple[int, dict[str, int]]:
    """Return the total byte footprint and the per-table breakdown."""
    row_counts = get_row_counts(scale_factor)

    table_to_mem_usage: dict[str, int] = {}
    for table_name, columns in schema.column_orders().items():
        row_size = sum(_type_size(dt) for dt in schema.column_types(table_name, columns))
        table_to_mem_usage[table_name] = row_size * row_counts[table_name]

    return sum(table_to_mem_usage.values()), table_to_mem_usage


def estimate_memory(
    scale_factor: float, identifier_type: str, decimal_type: str = "double"
) -> None:
    """Print the total and per-table memory estimate for the whole dataset."""
    schema = TpchTableSchema(identifier_type=identifier_type, decimal_type=decimal_type)
    total, by_table = calculate_memory_requirements(schema, scale_factor)

    print(f"  Total memory needed: {total / _BYTES_PER_GIB:.2f} GiB")
    for table_name, mem_needed in by_table.items():
        print(f"  Table: {table_name}, memory needed: {mem_needed / _BYTES_PER_GIB:.3f} GiB")


def main() -> None:
    """Parse arguments and print the estimate."""
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
        "--decimal-type",
        help="Representation for DECIMAL columns. 'double' (FLOAT64) or 'decimal' (DECIMAL64).",
        choices=["double", "decimal"],
        default="double",
    )

    args = arg_parser.parse_args()
    estimate_memory(args.scale_factor, args.identifier_type, args.decimal_type)


if __name__ == "__main__":
    main()
