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
    """Build the physical plan for TPC-H Q10 (returned item reporting)."""
    orders = schema.read(
        "orders",
        ["o_orderkey", "o_custkey", "o_orderdate"],
        (CR(4) >= DateLiteral("1993-10-01")) & (CR(4) < DateLiteral("1994-01-01")),
    )

    orders = orders.filter(
        (CR(2) >= DateLiteral("1993-10-01")) & (CR(2) < DateLiteral("1994-01-01")),
        [0, 1],
    )

    customer = schema.read(
        "customer",
        ["c_custkey", "c_nationkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment"],
    )

    # j1: "o_orderkey", "c_custkey", "c_nationkey", "c_name", "c_acctbal",
    #     "c_phone", "c_address", "c_comment"
    j1 = orders.broadcast_join(
        customer,
        CR(1) == CR(2),
        [0, 2, 3, 4, 5, 6, 7, 8],
        "inner",
        True,
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    nation = schema.read("nation", ["n_nationkey", "n_name"])

    # j2: "n_name", "o_orderkey", "c_custkey", "c_name", "c_acctbal",
    #     "c_phone", "c_address", "c_comment"
    j2 = nation.broadcast_join(
        j1,
        CR(0) == CR(4),
        [1, 2, 3, 5, 6, 7, 8, 9],
        "inner",
        True,
        unique_keys_policy=UniqueKeysPolicy.LEFT,
        perfect_hashing=True,
    )

    lineitem = schema.read(
        "lineitem",
        ["l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"],
        CR(8) == Literal(ord("R")),
    )
    lineitem = lineitem.filter(CR(1) == Literal(ord("R")), [0, 2, 3])

    # j3: "l_extendedprice", "l_discount", "n_name", "c_custkey", "c_name",
    #     "c_acctbal", "c_phone", "c_address", "c_comment"
    j3 = lineitem.broadcast_join(
        j2,
        CR(0) == CR(4),
        [1, 2, 3, 5, 6, 7, 8, 9, 10],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # agg: "n_name", "c_custkey", "c_name", "c_acctbal", "c_phone",
    #      "c_address", "c_comment", "revenue"
    agg = j3.aggregate(
        [CR(2), CR(3), CR(4), CR(5), CR(6), CR(7), CR(8)],
        [("sum", CR(0) * (DecimalLiteral("1", schema.decimal_column_type) - CR(1)))],
        perfect_hashing=False,
    )

    sort_limit = agg.sort([(CR(7), "descending", "before")]).fetch(0, 20)

    return sort_limit.project([CR(1), CR(2), CR(7), CR(3), CR(0), CR(5), CR(4), CR(6)])
