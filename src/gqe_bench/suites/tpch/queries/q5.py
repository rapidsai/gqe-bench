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
    """Build the physical plan for TPC-H Q5 (local supplier volume)."""
    # read customer table
    customer = schema.read("customer", ["c_nationkey", "c_custkey"], None)

    # read orders table
    orders = schema.read(
        "orders",
        ["o_custkey", "o_orderkey", "o_orderdate"],
        (CR(4) >= DateLiteral("1994-01-01")) & (CR(4) <= DateLiteral("1994-12-31")),
    )
    orders = orders.filter(
        (CR(2) >= DateLiteral("1994-01-01")) & (CR(2) <= DateLiteral("1994-12-31")),
        [0, 1],
    )
    # orders has ["o_custkey", "o_orderkey"]

    # broadcast join - customer is smaller
    result = orders.broadcast_join(
        customer,
        CR(0) == CR(3),
        [1, 2],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )
    # result has ["o_orderkey", "c_nationkey"]

    # broadcast join - lineitem result is smaller
    lineitem = schema.read(
        "lineitem",
        ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"],
        None,
    )
    result = lineitem.broadcast_join(
        result,
        CR(0) == CR(4),
        [1, 2, 3, 5],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )
    # result has ["l_suppkey", "l_extendedprice", "l_discount", "c_nationkey"]

    # broadcast join - supplier is smaller
    supplier = schema.read("supplier", ["s_suppkey", "s_nationkey"], None)
    result = result.broadcast_join(
        supplier,
        (CR(0) == CR(4)) & (CR(3) == CR(5)),
        [1, 2, 5],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )
    # result has ["l_extendedprice","l_discount", "s_nationkey"]

    # broadcast join nation
    nation = schema.read("nation", ["n_nationkey", "n_regionkey", "n_name"], None)
    result = result.broadcast_join(
        nation,
        (CR(2) == CR(3)),
        [0, 1, 4, 5],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )
    # result has [ "l_extendedprice", "l_discount", "n_regionkey", "n_name"]

    # broadcast join -  after filter - region
    region = schema.read("region", ["r_regionkey", "r_name"], None).filter(
        CR(1) == Literal("ASIA"), [0]
    )
    result = result.broadcast_join(
        region,
        (CR(2) == CR(4)),
        [0, 1, 3],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )
    # result has [ "l_extendedprice", "l_discount",  "n_name"]

    # groupby and sort
    result = result.aggregate(
        [CR(2)],
        [("sum", CR(0) * (DecimalLiteral("1", schema.decimal_column_type) - CR(1)))],
        perfect_hashing=False,
    )
    return result.sort([(CR(1), "descending", "before")])
