# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H Q21 (suppliers who kept orders waiting).

select
        s_name,
        count(*) as numwait
from
        supplier,
        lineitem l1,
        orders,
        nation
where
        s_suppkey = l1.l_suppkey
        and o_orderkey = l1.l_orderkey
        and o_orderstatus = 'F'
        and l1.l_receiptdate > l1.l_commitdate
        and exists (
                select
                        *
                from
                        lineitem l2
                where
                        l2.l_orderkey = l1.l_orderkey
                        and l2.l_suppkey <> l1.l_suppkey
        )
        and not exists (
                select
                        *
                from
                        lineitem l3
                where
                        l3.l_orderkey = l1.l_orderkey
                        and l3.l_suppkey <> l1.l_suppkey
                        and l3.l_receiptdate > l3.l_commitdate
        )
        and s_nationkey = n_nationkey
        and n_name = 'SAUDI ARABIA'
group by
        s_name
order by
        numwait desc,
        s_name
limit
        100
"""

from gqe_bench.physical_plan.expression import ColumnReference as CR
from gqe_bench.physical_plan.expression import Literal
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def _build_supplier(schema: TpchTableSchema) -> Relation:
    """Supplier filtered by nation = 'SAUDI ARABIA'."""
    nation = schema.read(
        "nation",
        ["n_nationkey", "n_name"],
        CR(1) == Literal("SAUDI ARABIA"),
    ).filter(CR(1) == Literal("SAUDI ARABIA"), [0])

    return schema.read(
        "supplier",
        ["s_suppkey", "s_name", "s_nationkey"],
        None,
    ).broadcast_join(
        nation,
        CR(2) == CR(3),
        [0, 1],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q21 (suppliers who kept orders waiting)."""
    supplier = _build_supplier(schema)

    # l1.l_receiptdate > l1.l_commitdate
    # `l1` has columns ["l_suppkey", "l_orderkey"]
    l1 = schema.read(
        "lineitem",
        ["l_suppkey", "l_orderkey", "l_receiptdate", "l_commitdate"],
        CR(12) > CR(11),
    ).filter(CR(2) > CR(3), [0, 1])

    # s_suppkey = l1.l_suppkey
    # `l1` has columns ["l_suppkey", "l_orderkey", "s_name"]
    l1 = l1.broadcast_join(
        supplier,
        CR(0) == CR(2),
        [0, 1, 3],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # l3.l_receiptdate > l3.l_commitdate
    # `l3` has columns ["l_suppkey", "l_orderkey"]
    l3 = schema.read(
        "lineitem",
        ["l_suppkey", "l_orderkey", "l_receiptdate", "l_commitdate"],
        CR(12) > CR(11),
    ).filter(CR(2) > CR(3), [0, 1])

    # not exists (l3.l_orderkey = l1.l_orderkey and l3.l_suppkey <> l1.l_suppkey)
    l1 = l1.broadcast_join(
        l3,
        (CR(1) == CR(4)) & (CR(0) != CR(3)),
        [0, 1, 2],
        "left_anti",
        True,
    )

    # o_orderkey = l1.l_orderkey and o_orderstatus = 'F'
    order = schema.read("orders", ["o_orderkey", "o_orderstatus"], CR(2) == Literal(70))
    order = order.filter(CR(1) == Literal(70), [0])
    l1 = order.broadcast_join(
        l1,
        CR(0) == CR(2),
        [1, 2, 3],
        unique_keys_policy=UniqueKeysPolicy.LEFT,
        perfect_hashing=True,
    )

    # exists (l2.l_orderkey = l1.l_orderkey and l2.l_suppkey <> l1.l_suppkey)
    l2 = schema.read("lineitem", ["l_suppkey", "l_orderkey"], None)
    l1 = l1.broadcast_join(l2, (CR(1) == CR(4)) & (CR(0) != CR(3)), [2], "left_semi", True)

    # group by s_name, order by numwait desc, s_name, limit 100
    l1 = (
        l1.aggregate([CR(0)], [("count_all", CR(0))], perfect_hashing=False)
        .sort([(CR(1), "descending", "before"), (CR(0), "ascending", "before")])
        .fetch(0, 100)
    )

    return l1
