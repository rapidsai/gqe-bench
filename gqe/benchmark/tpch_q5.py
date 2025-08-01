# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import read
from gqe.expression import ColumnReference as CR
from gqe.expression import Literal, DateLiteral
from gqe.benchmark.query import Query
from gqe.lib import UniqueKeysPolicy


"""
-- TPC-H Query 5
select
        n_name,
        sum(l_extendedprice * (1 - l_discount)) as revenue
from
        customer,
        orders,
        lineitem,
        supplier,
        nation,
        region
where
        c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and l_suppkey = s_suppkey
        and c_nationkey = s_nationkey
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = 'ASIA'
        and o_orderdate >= date '1994-01-01'
        and o_orderdate < date '1994-01-01' + interval '1' year
group by
        n_name
order by
        revenue desc
"""


class tpch_q5(Query):
    def root_relation(self):
        # read customer table
        customer = read("customer", ["c_nationkey", "c_custkey"])

        # read orders table
        orders = read("orders", ["o_custkey", "o_orderkey", "o_orderdate"], (CR(4) >= DateLiteral("1994-01-01")) & (CR(4) <= DateLiteral("1994-12-31")))
        orders = orders.filter(
            (CR(2) >= DateLiteral("1994-01-01")) & (CR(2) <= DateLiteral("1994-12-31")),
            [0, 1],
        )
        # orders has ["o_custkey", "o_orderkey"]

        # broadcast join - customer is smaller
        result = orders.broadcast_join(customer, CR(0) == CR(3), [1, 2], unique_keys_policy=UniqueKeysPolicy.right, perfect_hashing=True)
        # result has ["o_orderkey", "c_nationkey"]

        # broadcast join - lineitem result is smaller
        lineitem = read(
            "lineitem", ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]
        )
        result = lineitem.broadcast_join(result, CR(0) == CR(4), [1, 2, 3, 5], unique_keys_policy=UniqueKeysPolicy.right, perfect_hashing=True)
        # result has ["l_suppkey", "l_extendedprice", "l_discount", "c_nationkey"]

        # broadcast join - supplier is smaller
        supplier = read("supplier", ["s_suppkey", "s_nationkey"])
        result = result.broadcast_join(
            supplier, (CR(0) == CR(4)) & (CR(3) == CR(5)), [1, 2, 5], unique_keys_policy=UniqueKeysPolicy.right, perfect_hashing=True
        )
        # result has ["l_extendedprice","l_discount", "s_nationkey"]

        # broadcast join nation
        nation = read("nation", ["n_nationkey", "n_regionkey", "n_name"])
        result = result.broadcast_join(nation, (CR(2) == CR(3)), [0, 1, 4, 5], unique_keys_policy=UniqueKeysPolicy.right, perfect_hashing=True)
        # result has [ "l_extendedprice", "l_discount", "n_regionkey", "n_name"]

        # broadcast join -  after filter - region
        region = read("region", ["r_regionkey", "r_name"]).filter(
            CR(1) == Literal("ASIA"), [0]
        )
        result = result.broadcast_join(region, (CR(2) == CR(4)), [0, 1, 3], unique_keys_policy=UniqueKeysPolicy.right, perfect_hashing=True)
        # result has [ "l_extendedprice", "l_discount",  "n_name"]

        # groupby and sort
        result = result.aggregate([CR(2)], [("sum", CR(0) * (Literal(1) - CR(1)))])
        return result.sort([(CR(1), "descending", "before")])
