# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.expression import (
    DateLiteral,
    DecimalLiteral,
)
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q15 (top supplier)."""
    lineitem = schema.read(
        "lineitem",
        ["l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"],
        (CR(10) >= DateLiteral("1996-01-01")) & (CR(10) <= DateLiteral("1996-03-31")),
    ).filter(
        (CR(1) >= DateLiteral("1996-01-01")) & (CR(1) <= DateLiteral("1996-03-31")),
        [0, 2, 3],
    )

    revenue = lineitem.aggregate(
        [CR(0)],
        [("sum", CR(1) * (DecimalLiteral("1", schema.decimal_column_type) - CR(2)))],
        perfect_hashing=True,
    )

    max_revenue = revenue.aggregate([], [("max", CR(1))], perfect_hashing=True)

    l_max_revenue = revenue.broadcast_join(
        max_revenue,
        CR(1) == CR(2),
        [0, 1],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    supplier = schema.read(
        "supplier",
        ["s_suppkey", "s_name", "s_address", "s_phone"],
    )

    unsorted_output = supplier.broadcast_join(
        l_max_revenue,
        CR(0) == CR(4),
        [0, 1, 2, 3, 5],
        unique_keys_policy=UniqueKeysPolicy.LEFT,
        perfect_hashing=True,
    )

    return unsorted_output.sort([(CR(0), "ascending", "before")])
