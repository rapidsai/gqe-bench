# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.expression import (
    LikeExpr,
    Literal,
)
from gqe_bench.physical_plan.relation import Relation
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q13 (customer distribution)."""
    customer = schema.read("customer", ["c_custkey"])

    orders = schema.read("orders", ["o_custkey", "o_comment"]).filter(
        LikeExpr(CR(1), "%special%requests%") == Literal(False), [0]
    )

    # Generate a dummy o_orderkey column for the left outer join
    orders = orders.project([Literal(1), CR(0)])

    customer_orders = customer.broadcast_join(orders, CR(0) == CR(2), [0, 1], "left")

    grouped_customer_orders = customer_orders.aggregate([CR(0)], [("count_valid", CR(1))])

    grouped_c_count = grouped_customer_orders.aggregate([CR(1)], [("count_all", CR(1))])

    return grouped_c_count.sort(
        [
            (CR(1), "descending", "before"),
            (CR(0), "descending", "before"),
        ]
    )
