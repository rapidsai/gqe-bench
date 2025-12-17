# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import read
from gqe.benchmark.query import Query
from gqe.expression import ColumnReference as CR
from gqe.expression import DateLiteral, Literal
from gqe.lib import UniqueKeysPolicy
from gqe.table_definition import TPCHTableDefinitions

"""
select
        c_custkey,
        c_name,
        sum(l_extendedprice * (1 - l_discount)) as revenue,
        c_acctbal,
        n_name,
        c_address,
        c_phone,
        c_comment
from
        customer,
        orders,
        lineitem,
        nation
where
        c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and o_orderdate >= date '1993-10-01'
        and o_orderdate < date '1993-10-01' + interval '3' month
        and l_returnflag = 'R'
        and c_nationkey = n_nationkey
group by
        c_custkey,
        c_name,
        c_acctbal,
        c_phone,
        n_name,
        c_address,
        c_comment
order by
        revenue desc
limit
        20
"""


# Plan improves on the substrait plan by having a better join order and reducing string materialization cost.
class tpch_q10_fused_filter(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        orders = read(
            "orders",
            ["o_orderkey", "o_custkey", "o_orderdate"],
            (CR(4) >= DateLiteral("1993-10-01")) & (CR(4) < DateLiteral("1994-01-01")),
            table_defs,
        )

        customer = read(
            "customer",
            [
                "c_custkey",
                "c_nationkey",
                "c_name",
                "c_acctbal",
                "c_phone",
                "c_address",
                "c_comment",
            ],
            None,
            table_defs,
        )

        # `orders` has filter o_orderdate >= date '1993-10-01' and o_orderdate < date '1993-10-01' + interval '3' month
        # c_custkey = o_custkey
        # broadcast orders table
        # j1 has columns ["o_orderkey", "c_custkey", "c_nationkey", "c_name", "c_acctbal", "c_phone","c_address", "c_comment"]
        j1 = orders.broadcast_join(
            customer,
            CR(1) == CR(3),
            [0, 3, 4, 5, 6, 7, 8, 9],
            "inner",
            broadcast_left=False,  # FIXME: should be benchmarked whether to use broadcast_left or broadcast_right.
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
            left_filter=(
                (CR(2) >= DateLiteral("1993-10-01")) & (CR(2) < DateLiteral("1994-01-01"))
            ),
        )

        nation = read("nation", ["n_nationkey", "n_name"], None, table_defs)

        # c_nationkey = n_nationkey
        # broadcast nation table
        # j2 has columns ["n_name", "o_orderkey", "c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment"]
        j2 = nation.broadcast_join(
            j1,
            CR(0) == CR(4),
            [1, 2, 3, 5, 6, 7, 8, 9],
            "inner",
            True,
            unique_keys_policy=UniqueKeysPolicy.left,
            perfect_hashing=True,
        )

        lineitem = read(
            "lineitem",
            ["l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"],
            (CR(8) == Literal(ord("R"))),
            table_defs,
        )

        # `lineitem` has filter l_returnflag = 'R'
        # l_orderkey = o_orderkey
        # broadcast right side (join)
        # j3 hash columns ["l_extendedprice", "l_discount", "n_name", "c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment"]
        j3 = lineitem.broadcast_join(
            j2,
            CR(0) == CR(5),
            [2, 3, 4, 6, 7, 8, 9, 10, 11],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
            left_filter=(CR(1) == Literal(ord("R"))),
        )

        # group by "n_name", "c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment"
        # sum l_extendedprice * (1 - l_discount)
        # agg has "n_name", "c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "revenue"
        agg = j3.aggregate(
            [CR(2), CR(3), CR(4), CR(5), CR(6), CR(7), CR(8)],
            [("sum", CR(0) * (Literal(1.0) - CR(1)))],
            perfect_hashing=False,
        )

        # order by "revenue" desc limit 20
        sort_limit = agg.sort([(CR(7), "descending", "before")]).fetch(0, 20)

        # Project
        # c_custkey,
        # c_name,
        # sum(l_extendedprice * (1 - l_discount)) as revenue,
        # c_acctbal,
        # n_name,
        # c_address,
        # c_phone,
        # c_comment
        return sort_limit.project([CR(1), CR(2), CR(7), CR(3), CR(0), CR(5), CR(4), CR(6)])
