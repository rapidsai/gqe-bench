# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.expression import (
    DateLiteral,
    DecimalLiteral,
    Literal,
)
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q3 (shipping priority)."""
    customer = schema.read(
        "customer",
        ["c_custkey", "c_mktsegment"],
        (CR(6) == Literal("BUILDING")),
    )
    orders = schema.read(
        "orders",
        ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
        (CR(4) < DateLiteral("1995-03-15")),
    )
    lineitem = schema.read(
        "lineitem",
        ["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"],
        (CR(10) > DateLiteral("1995-03-15")),
    )

    # Filter customer table: c_mktsegment = 'BUILDING'
    # filtered_customer: c_custkey(0)
    filtered_customer = customer.filter(CR(1) == Literal("BUILDING"), [0])

    # Filter orders table: o_orderdate < date '1995-03-15'
    # filtered_orders: o_orderkey, o_orderdate, o_shippriority
    filtered_orders = orders.filter(CR(2) < DateLiteral("1995-03-15"), [0, 1, 2, 3])

    # Join customer and orders tables: c_custkey = o_custkey
    # orders_customer: o_orderkey, o_orderdate, o_shippriority
    orders_customer = filtered_orders.broadcast_join(
        filtered_customer,
        CR(1) == CR(4),
        [0, 2, 3],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # Filter lineitem table: l_shipdate > date '1995-03-15'
    # filtered_lineitem: l_orderkey, l_extendedprice, l_discount
    filtered_lineitem = lineitem.filter(
        CR(3) > DateLiteral("1995-03-15"),
        [0, 1, 2],
    )

    # Join orders and lineitem tables: o_orderkey = l_orderkey
    # After join indices: l_orderkey, o_orderdate, o_shippriority, l_extendedprice, l_discount
    joined = filtered_lineitem.broadcast_join(
        orders_customer,
        CR(0) == CR(3),
        [0, 4, 5, 1, 2],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # Group by: l_orderkey, o_orderdate, o_shippriority
    # Aggregate function: sum(l_extendedprice * (1 - l_discount)) as revenue
    # After aggregation: l_orderkey, o_orderdate, o_shippriority, revenue
    aggregated = joined.aggregate(
        [CR(0), CR(1), CR(2)],
        [("sum", CR(3) * (DecimalLiteral("1", schema.decimal_column_type) - CR(4)))],
        perfect_hashing=True,
    )

    # Sort by: revenue DESC, o_orderdate ASC
    sorted_result = aggregated.sort(
        [
            (CR(3), "descending", "after"),
            (CR(1), "ascending", "after"),
        ]
    )

    # Fetch top 10 rows
    fetched = sorted_result.fetch(0, 10)

    # Project to match SQL output column order: l_orderkey, revenue, o_orderdate, o_shippriority
    return fetched.project([CR(0), CR(3), CR(1), CR(2)])
