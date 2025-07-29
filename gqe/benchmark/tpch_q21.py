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
from gqe.expression import ColumnReference as CR, Literal
from gqe.benchmark.query import Query
from gqe.lib import UniqueKeysPolicy

"""
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


class tpch_q21(Query):
    def root_relation(self):
        # `supplier` has columns ["s_suppkey", "s_name"] which satisfies
        # s_nationkey = n_nationkey and n_name = 'SAUDI ARABIA'
        nation = read("nation", ["n_nationkey", "n_name"], CR(1) == Literal("SAUDI ARABIA")).filter(
            CR(1) == Literal("SAUDI ARABIA"), [0]
        )
        supplier = read(
            "supplier", ["s_suppkey", "s_name", "s_nationkey"]
        ).broadcast_join(nation, CR(2) == CR(3), [0, 1], unique_keys_policy=UniqueKeysPolicy.right, perfect_hashing=True)

        # l1.l_receiptdate > l1.l_commitdate
        # `l1` has columns ["l_suppkey", "l_orderkey"]
        l1 = read(
            "lineitem", ["l_suppkey", "l_orderkey", "l_receiptdate", "l_commitdate"],
            CR(12) > CR(11)
        ).filter(CR(2) > CR(3), [0, 1])

        # s_suppkey = l1.l_suppkey
        # `l1` has columns ["l_suppkey", "l_orderkey", "s_name"]
        l1 = l1.broadcast_join(supplier, CR(0) == CR(2), [0, 1, 3], unique_keys_policy=UniqueKeysPolicy.right, perfect_hashing=True)

        # l3.l_receiptdate > l3.l_commitdate
        # `l3` has columns ["l_suppkey", "l_orderkey"]
        l3 = read(
            "lineitem", ["l_suppkey", "l_orderkey", "l_receiptdate", "l_commitdate"],
            CR(12) > CR(11)
        ).filter(CR(2) > CR(3), [0, 1])

        # l1 has columns ["l_suppkey", "l_orderkey", "s_name"] which satisfies
        # not exists (
        #    select * from
        #    lineitem l3
        #    where
        #        l3.l_orderkey = l1.l_orderkey
        #        and l3.l_suppkey <> l1.l_suppkey
        #        and l3.l_receiptdate > l3.l_commitdate
        # )
        l1 = l1.broadcast_join(
            l3, (CR(1) == CR(4)) & (CR(0) != CR(3)), [0, 1, 2], "left_anti", True)

        # o_orderkey = l1.l_orderkey and o_orderstatus = 'F'
        order = read("orders", ["o_orderkey", "o_orderstatus"], CR(2) == Literal(70))
        order = order.filter(CR(1) == Literal(70), [0])
        l1 = order.broadcast_join(l1, CR(0) == CR(2), [1, 2, 3], unique_keys_policy=UniqueKeysPolicy.left, perfect_hashing=True)

        # l1 has columns ["l_suppkey", "l_orderkey", "s_name"] which satisfies
        # exists (
        #     select * from
        #     lineitem l2
        #     where
        #         l2.l_orderkey = l1.l_orderkey
        #         and l2.l_suppkey <> l1.l_suppkey
        # )
        l2 = read("lineitem", ["l_suppkey", "l_orderkey"])
        l1 = l1.broadcast_join(
            l2, (CR(1) == CR(4)) & (CR(0) != CR(3)), [2], "left_semi", True)

        # group by
        #     s_name
        # order by
        #     numwait desc, s_name
        # limit 100
        l1 = (
            l1.aggregate([CR(0)], [("count_all", CR(0))])
            .sort([(CR(1), "descending", "before"), (CR(0), "ascending", "before")])
            .fetch(0, 100)
        )

        return l1
