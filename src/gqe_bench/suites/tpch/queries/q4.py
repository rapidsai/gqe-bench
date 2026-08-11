# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.expression import (
    DateLiteral,
)
from gqe_bench.physical_plan.relation import Relation
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q4 (order priority checking)."""
    orders = schema.read(
        "orders",
        ["o_orderkey", "o_orderdate", "o_orderpriority"],
        (CR(4) >= DateLiteral("1993-07-01")) & (CR(4) < DateLiteral("1993-10-01")),
    )

    # o_orderdate >= date '1993-07-01' and o_orderdate < date '1993-07-01' + interval '3' month
    # After this operation, `orders` has column ["o_orderkey", "o_orderpriority"]
    orders = orders.filter(
        (CR(1) >= DateLiteral("1993-07-01")) & (CR(1) < DateLiteral("1993-10-01")),
        [0, 2],
    )

    # l_commitdate < l_receiptdate
    # After this operation, `lineitem` has column ["l_orderkey"]
    lineitem = schema.read(
        "lineitem",
        ["l_orderkey", "l_commitdate", "l_receiptdate"],
        (CR(11) < CR(12)),
    )
    lineitem = lineitem.filter(CR(1) < CR(2), [0])

    # exists (select * from lineitem where l_orderkey = o_orderkey)
    # Broadcast the left side i.e. `orders` table
    # After this operation, `orders` has column ["o_orderpriority"]
    orders = orders.broadcast_join(lineitem, CR(0) == CR(2), [1], "left_semi", True)

    # group by o_orderpriority
    orders = orders.aggregate([CR(0)], [("count_all", CR(0))], perfect_hashing=False)

    # order by o_orderpriority
    return orders.sort([(CR(0), "ascending", "before")])
