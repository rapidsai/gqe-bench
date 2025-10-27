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
from gqe.expression import ColumnReference as CR
from gqe.expression import DateLiteral
from gqe.expression import Literal
from gqe.benchmark.query import Query
from gqe.benchmark.hardcoded.bindings.tpch_q10 import (
    Q10FusedProbesJoinMapBuildRelation,
    Q10FusedProbesJoinMultimapBuildRelation,
    Q10FusedProbesJoinProbeRelation,
    Q10SortLimitRelation,
    Q10UniqueKeyInnerJoinBuildRelation,
    Q10UniqueKeyInnerJoinProbeRelation,
)
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


# Optimizations vs. baseline of hand-coded Q10 plan:
#   - Modified join order.
#   - Building multimap on `o_custkey` instead of map on `c_custkey` to reduce
#     build size and enable Bloom filter.
#   - Fusing probes of j2 and j3 (in reference to hand-coded plan) to reduce
#     intermediate materialization.
#   - Combined `order by` and `limit` clauses into a sort-limit relation to
#     reduce intermediate materialization.
class tpch_q10_opt(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        orders = read(
            "orders",
            ["o_orderkey", "o_custkey", "o_orderdate"],
            (CR(4) >= DateLiteral("1993-10-01")) & (CR(4) < DateLiteral("1994-01-01")),
            table_defs,
        )

        # o_orderdate >= date '1993-10-01' and o_orderdate < date '1993-10-01' + interval '3' month
        # After this operation, `orders` has column ["o_orderkey", "o_custkey"]
        orders = orders.filter(
            (CR(2) >= DateLiteral("1993-10-01")) & (CR(2) < DateLiteral("1994-01-01")),
            [0, 1],
        )

        lineitem = read(
            "lineitem",
            ["l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"],
            (CR(8) == Literal(ord("R"))),
            table_defs,
        )

        # o_orderkey = l_orderkey
        # hardcoded probe filter: l_returnflag = 'R'
        # j1 has columns ["o_custkey", "l_extendedprice", "l_discount"]
        orders_map = Q10UniqueKeyInnerJoinBuildRelation(orders, 0, True)
        j1 = Q10UniqueKeyInnerJoinProbeRelation(
            orders_map, orders, lineitem, 0, 0, [1, 4, 5]
        )

        # build hash multimap on o_custkey in j1
        j1_multimap = Q10FusedProbesJoinMultimapBuildRelation(j1, 0)

        nation = read("nation", ["n_nationkey", "n_name"], None, table_defs)
        # build hash map on n_nationkey in nation_map
        nation_map = Q10FusedProbesJoinMapBuildRelation(nation, 0)

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
        # probe j1 on c_custkey = o_custkey, then probe nation on c_nationkey = n_nationkey
        # j3 (so named since j2 is fused together) has columns ["l_extendedprice", "l_discount", "n_name", "c_custkey", "c_name", "c_acctbal", "c_phone","c_address", "c_comment"]
        j3 = Q10FusedProbesJoinProbeRelation(
            j1_multimap, nation_map, j1, nation, customer
        )

        # group by "n_name", "c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment"
        # sum l_extendedprice * (1 - l_discount)
        # agg has "n_name", "c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "revenue"
        agg = j3.aggregate(
            [CR(2), CR(3), CR(4), CR(5), CR(6), CR(7), CR(8)],
            [("sum", CR(0) * (Literal(1.0) - CR(1)))],
        )

        # order by "revenue" desc limit 20
        # Project:
        #   c_custkey,
        #   c_name,
        #   sum(l_extendedprice * (1 - l_discount)) as revenue,
        #   c_acctbal,
        #   n_name,
        #   c_address,
        #   c_phone,
        #   c_comment
        return Q10SortLimitRelation(agg, 7, 20, [1, 2, 7, 3, 0, 5, 4, 6])
