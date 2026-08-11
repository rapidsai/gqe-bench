# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H Q18 (large volume customer).

select
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice,
        sum(l_quantity)
from
        customer,
        orders,
        lineitem
where
        o_orderkey in (
                select
                        l_orderkey
                from
                        lineitem
                group by
                        l_orderkey having
                                sum(l_quantity) > 300
        )
        and c_custkey = o_custkey
        and o_orderkey = l_orderkey
group by
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice
order by
        o_totalprice desc,
        o_orderdate
limit
        100
"""

from gqe_bench.physical_plan.expression import ColumnReference as CR
from gqe_bench.physical_plan.expression import DecimalLiteral
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q18 (large volume customer)."""
    # After this operation, `lineitem` contains [l_orderkey, sum(l_quantity)]
    lineitem = (
        schema.read("lineitem", ["l_orderkey", "l_quantity"], None)
        .aggregate([CR(0)], [("sum", CR(1))], perfect_hashing=True)
        .filter(CR(1) > DecimalLiteral("300", schema.decimal_column_type), [0, 1])
    )

    customer = schema.read("customer", ["c_custkey", "c_name"], None)

    # After this operation, `orders` contains
    # [o_orderkey, o_custkey, o_orderdate, o_totalprice, sum(l_quantity)]
    orders = schema.read(
        "orders",
        ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"],
        None,
    ).broadcast_join(
        lineitem,
        CR(0) == CR(4),
        [0, 1, 2, 3, 5],
        unique_keys_policy=UniqueKeysPolicy.LEFT,
        perfect_hashing=True,
    )

    # After this operation, `orders` contains
    # [o_orderkey, c_custkey, o_orderdate, o_totalprice, sum(l_quantity), c_name]
    orders = orders.broadcast_join(
        customer,
        CR(1) == CR(5),
        [0, 1, 2, 3, 4, 6],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # After this operation, `orders` contains
    # [c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)]
    orders = (
        orders.aggregate(
            [CR(5), CR(1), CR(0), CR(2), CR(3)],
            [("sum", CR(4))],
            perfect_hashing=False,
        )
        .sort([(CR(4), "descending", "before"), (CR(3), "ascending", "before")])
        .fetch(0, 100)
    )

    return orders
