# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.expression import (
    DateLiteral,
    IfThenElseExpr,
    Literal,
)
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q12 (shipping modes and order priority)."""
    lineitem = schema.read(
        "lineitem",
        ["l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate", "l_orderkey"],
        ((CR(14) == Literal("MAIL")) | (CR(14) == Literal("SHIP")))
        & (CR(12) > CR(11))
        & (CR(11) > CR(10))
        & (CR(12) >= DateLiteral("1994-01-01"))
        & (CR(12) <= DateLiteral("1994-12-31")),
    )

    lineitem = lineitem.filter(
        ((CR(0) == Literal("MAIL")) | (CR(0) == Literal("SHIP")))
        & (CR(1) < CR(2))
        & (CR(3) < CR(1))
        & (CR(2) >= DateLiteral("1994-01-01"))
        & (CR(2) <= DateLiteral("1994-12-31")),
        [0, 4],
    )

    orders = schema.read("orders", ["o_orderkey", "o_orderpriority"])

    join_out = orders.broadcast_join(
        lineitem,
        CR(0) == CR(3),
        [1, 2],
        unique_keys_policy=UniqueKeysPolicy.LEFT,
        perfect_hashing=True,
    )

    agg_out = join_out.aggregate(
        [CR(1)],
        [
            (
                "sum",
                IfThenElseExpr(
                    (CR(0) == Literal("1-URGENT")) | (CR(0) == Literal("2-HIGH")),
                    Literal(1),
                    Literal(0),
                ),
            ),
            (
                "sum",
                IfThenElseExpr(
                    (CR(0) != Literal("1-URGENT")) & (CR(0) != Literal("2-HIGH")),
                    Literal(1),
                    Literal(0),
                ),
            ),
        ],
        perfect_hashing=False,
    )

    return agg_out.sort([(CR(0), "ascending", "before")])
