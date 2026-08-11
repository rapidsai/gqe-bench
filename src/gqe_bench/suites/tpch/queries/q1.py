# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    Cast,
    DateLiteral,
    DecimalLiteral,
)
from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.relation import Relation
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q1 (pricing summary report)."""
    lineitem = schema.read(
        "lineitem",
        [
            "l_shipdate",
            "l_discount",
            "l_quantity",
            "l_extendedprice",
            "l_returnflag",
            "l_linestatus",
            "l_tax",
        ],
        CR(10) <= DateLiteral("1998-09-02"),
    )

    one = DecimalLiteral("1", schema.decimal_column_type)

    if schema.is_fixed_point:
        decimal128_type = schema.decimal128_type()
        lineitem = lineitem.project(
            [
                CR(0),
                Cast(CR(1), decimal128_type),
                Cast(CR(2), decimal128_type),
                Cast(CR(3), decimal128_type),
                CR(4),
                CR(5),
                Cast(CR(6), decimal128_type),
            ]
        )

    agg = lineitem.aggregate(
        [CR(4), CR(5)],
        [
            ("sum", CR(2)),
            ("sum", CR(3)),
            ("sum", CR(3) * (one - CR(1))),
            ("sum", CR(3) * (one - CR(1)) * (one + CR(6))),
            ("sum", CR(1)),
            ("count_all", CR(1)),
        ],
        CR(0) <= DateLiteral("1998-09-02"),
        perfect_hashing=True,
    )

    project = agg.project(
        [
            CR(0),
            CR(1),
            CR(2),
            CR(3),
            CR(4),
            CR(5),
            CR(2) / CR(7),
            CR(3) / CR(7),
            CR(6) / CR(7),
            CR(7),
        ]
    )

    return project.sort(
        [
            (CR(0), "ascending", "before"),
            (CR(1), "ascending", "before"),
        ]
    )
