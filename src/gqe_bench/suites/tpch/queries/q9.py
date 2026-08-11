# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.expression import (
    DatePartExpr,
    DecimalLiteral,
    LikeExpr,
)
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q9 (product type profit measure)."""
    partsupp = schema.read("partsupp", ["ps_suppkey", "ps_partkey", "ps_supplycost"])

    part = schema.read("part", ["p_partkey", "p_name"]).filter(LikeExpr(CR(1), "%green%"), [0])

    # join1: "ps_suppkey", "ps_supplycost", "p_partkey"
    join1 = partsupp.broadcast_join(
        part,
        CR(1) == CR(3),
        [0, 2, 3],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    supplier = schema.read("supplier", ["s_suppkey", "s_nationkey"])

    # join2: "ps_supplycost", "p_partkey", "s_suppkey", "s_nationkey"
    join2 = join1.broadcast_join(
        supplier,
        CR(0) == CR(3),
        [1, 2, 3, 4],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    lineitem = schema.read(
        "lineitem",
        ["l_suppkey", "l_partkey", "l_orderkey", "l_extendedprice", "l_discount", "l_quantity"],
    )

    # join3: "l_orderkey", "l_extendedprice", "l_discount", "l_quantity",
    #        "ps_supplycost", "s_nationkey"
    join3 = lineitem.broadcast_join(
        join2,
        (CR(0) == CR(8)) & (CR(1) == CR(7)),
        [2, 3, 4, 5, 6, 9],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # "l_orderkey", "amount", "s_nationkey"
    join3_projected = join3.project(
        [
            CR(0),
            CR(1) * DecimalLiteral("1", schema.decimal_column_type) - CR(1) * CR(2) - CR(4) * CR(3),
            CR(5),
        ]
    )

    orders = schema.read("orders", ["o_orderkey", "o_orderdate"])

    # "o_orderdate", "amount", "s_nationkey"
    join4 = orders.broadcast_join(
        join3_projected,
        CR(0) == CR(2),
        [1, 3, 4],
        unique_keys_policy=UniqueKeysPolicy.LEFT,
        perfect_hashing=True,
    )

    nation = schema.read("nation", ["n_nationkey", "n_name"])

    # "n_name", "o_orderdate", "amount"
    join5 = join4.broadcast_join(
        nation,
        CR(2) == CR(3),
        [4, 0, 1],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    agg = join5.aggregate(
        [CR(0), DatePartExpr(CR(1), "year")],
        [("sum", CR(2))],
        perfect_hashing=False,
    )

    return agg.sort(
        [
            (CR(0), "ascending", "before"),
            (CR(1), "descending", "before"),
        ]
    )
