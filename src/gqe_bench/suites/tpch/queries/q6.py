# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.expression import (
    DateLiteral,
    DecimalLiteral,
)
from gqe_bench.physical_plan.relation import Relation
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q6 (forecasting revenue change)."""
    dec_type = schema.decimal_column_type
    lit_05 = DecimalLiteral("0.05", dec_type)
    lit_07 = DecimalLiteral("0.07", dec_type)
    lit_24 = DecimalLiteral("24", dec_type)

    lineitem = schema.read(
        "lineitem",
        ["l_shipdate", "l_discount", "l_quantity", "l_extendedprice"],
        (CR(10) >= DateLiteral("1994-01-01"))
        & (CR(10) < DateLiteral("1995-01-01"))
        & (CR(6) >= lit_05)
        & (CR(6) <= lit_07)
        & (CR(4) < lit_24),
    )

    # l_shipdate >= date '1994-01-01'
    # and l_shipdate < date '1994-01-01' + interval '1' year
    lineitem = lineitem.filter(
        (CR(0) >= DateLiteral("1994-01-01")) & (CR(0) < DateLiteral("1995-01-01")),
        [1, 2, 3],
    )

    # and l_discount between 0.06 - 0.01 and 0.06 + 0.01
    # and l_quantity < 24
    #
    # splitting the filter predicate effectively late materializes this
    # part when using zero-copy
    lineitem = lineitem.filter(
        (CR(0) >= lit_05) & (CR(0) <= lit_07) & (CR(1) < lit_24),
        [0, 2],
    )

    # sum(l_extendedprice * l_discount) as revenue
    return lineitem.aggregate([], [("sum", CR(1) * CR(0))], perfect_hashing=True)
