# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.expression import (
    DateLiteral,
    DatePartExpr,
    DecimalLiteral,
    Literal,
)
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q7 (volume shipping)."""
    # Nation filter predicate rewrite to a conjunctive normal form separable into multiple filters
    # P := (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY') or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
    # <=>
    # P := (n1.n_name = 'FRANCE' or n1.n_name = 'GERMANY') and (n2.n_name = 'GERMANY' or n2.n_name ='FRANCE') and n2.n_name /= n1.n_name

    # WHERE n1.n_name = 'FRANCE' or n1.n_name = 'GERMANY'
    nation = schema.read(
        "nation",
        ["n_nationkey", "n_name"],
        (CR(1) == Literal("FRANCE")) | (CR(1) == Literal("GERMANY")),
    ).filter((CR(1) == Literal("FRANCE")) | (CR(1) == Literal("GERMANY")), [0, 1])

    # join on c_nationkey = n2.n_nationkey
    #   customer has columns ["c_custkey", "c_nationkey"]
    #   returns ["c_custkey", "n_name" as "cust_nation"]
    customer = schema.read("customer", ["c_custkey", "c_nationkey"], None).broadcast_join(
        nation,
        CR(1) == CR(2),
        [0, 3],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # join on c_custkey = o_custkey
    #   orders has columns ["o_orderkey", "o_custkey"]
    #   customer has columns ["c_custkey", "cust_nation"]
    #   returns ["o_orderkey", "cust_nation"]
    orders = schema.read("orders", ["o_orderkey", "o_custkey"], None).broadcast_join(
        customer,
        CR(1) == CR(2),
        [0, 3],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # WHERE l_shipdate between date '1995-01-01' and date '1996-12-31'
    l1 = schema.read(
        "lineitem",
        ["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"],
        (CR(10) >= DateLiteral("1995-01-01")) & (CR(10) <= DateLiteral("1996-12-31")),
    ).filter(
        (CR(2) >= DateLiteral("1995-01-01")) & (CR(2) <= DateLiteral("1996-12-31")),
        [0, 1, 2, 3, 4],
    )

    # join on o_orderkey = l_orderkey
    #   l1 has columns ["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
    #   orders has columns ["o_orderkey", "cust_nation"]
    #   returns ["cust_nation", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
    l1 = l1.broadcast_join(
        orders,
        CR(0) == CR(5),
        [6, 1, 2, 3, 4],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # join on s_nationkey = n1.n_nationkey
    #   supplier has columns ["s_suppkey", "s_nationkey"]
    #   returns ["s_suppkey", "n_name" as "supp_nation"]
    supplier = schema.read("supplier", ["s_suppkey", "s_nationkey"], None).broadcast_join(
        nation,
        CR(1) == CR(2),
        [0, 3],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # join on s_suppkey = l_suppkey and n2.n_name /= n1.n_name
    #   l1 has columns ["cust_nation", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
    #   supplier has columns ["s_suppkey", "supp_nation"]
    #   returns ["supp_nation", "cust_nation", "l_shipdate", "l_extendedprice", "l_discount"]
    #   This is UniqueKeysPolicy.RIGHT but not marked in the join
    #   because we cannot yet do unique key joins on non-equijoins.
    l1 = l1.broadcast_join(supplier, (CR(1) == CR(5)) & (CR(0) != CR(6)), [6, 0, 2, 3, 4])

    # SELECT
    #   supp_nation,
    #   cust_nation,
    #   extract(year from l_shipdate) as l_year,
    #   l_extendedprice * (1 - l_discount) as volume
    l1 = l1.project(
        [
            CR(0),
            CR(1),
            DatePartExpr(CR(2), "year"),
            CR(3) * (DecimalLiteral("1", schema.decimal_column_type) - CR(4)),
        ]
    )

    # group by keys ["supp_nation", "cust_nation", "l_year"], aggs ["sum(volume)"]
    # order by supp_nation, cust_nation, l_year
    l1 = l1.aggregate([CR(0), CR(1), CR(2)], [("sum", CR(3))], perfect_hashing=False).sort(
        [
            (CR(0), "ascending", "before"),
            (CR(1), "ascending", "before"),
            (CR(2), "ascending", "before"),
        ]
    )

    return l1
