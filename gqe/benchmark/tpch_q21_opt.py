# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from gqe import read
from gqe.benchmark.hardcoded.bindings.tpch_q21 import (
    Q21LeftAntiJoinProbeRelation,
    Q21LeftAntiJoinRetrieveRelation,
    Q21LeftSemiJoinProbeRelation,
    Q21LeftSemiJoinRetrieveRelation,
)
from gqe.benchmark.query import Query
from gqe.expression import ColumnReference as CR
from gqe.expression import Literal
from gqe.lib import UniqueKeysPolicy
from gqe.table_definition import TPCHTableDefinitions

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


# Optimizations:
# - Fuse the lineitem filter and left anti join.
# - Use mark join to implement left anti join and left semi join.
# - Use bloom filter before join to reduce random access on hash table lookups.
class tpch_q21_opt(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # `supplier` has columns ["s_suppkey", "s_name"] which satisfies
        # s_nationkey = n_nationkey and n_name = 'SAUDI ARABIA'
        nation = read("nation", ["n_nationkey", "n_name"], CR(1) == Literal("SAUDI ARABIA")).filter(
            CR(1) == Literal("SAUDI ARABIA"), [0]
        )
        supplier = read(
            "supplier", ["s_suppkey", "s_name", "s_nationkey"], None, table_defs
        ).broadcast_join(
            nation,
            CR(2) == CR(3),
            [0, 1],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # l1.l_receiptdate > l1.l_commitdate
        # s_suppkey = l1.l_suppkey
        # `joined1` has columns ["l_suppkey", "l_orderkey", "s_name"]
        l1 = read(
            "lineitem",
            ["l_suppkey", "l_orderkey", "l_receiptdate", "l_commitdate"],
            CR(12) > CR(11),
            table_defs,
        ).filter(CR(2) > CR(3), [0, 1])
        # s_suppkey = l1.l_suppkey
        # `joined1` has columns ["l_suppkey", "l_orderkey", "s_name"]
        joined1 = l1.broadcast_join(
            supplier,
            CR(0) == CR(2),
            [0, 1, 3],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # joined2 has columns ["l_suppkey", "l_orderkey", "s_name"] which satisfies
        # not exists (
        #    select * from
        #    lineitem l3
        #    where
        #        l3.l_orderkey = l1.l_orderkey
        #        and l3.l_suppkey <> l1.l_suppkey
        #        and l3.l_receiptdate > l3.l_commitdate
        # )
        l2_probe = Q21LeftAntiJoinProbeRelation(
            joined1,
            read(
                "lineitem",
                ["l_suppkey", "l_orderkey", "l_receiptdate", "l_commitdate"],
                CR(12) > CR(11),
                table_defs,
            ),
        )
        joined2 = Q21LeftAntiJoinRetrieveRelation(joined1, l2_probe)

        # o_orderkey = l1.l_orderkey and o_orderstatus = 'F'
        order = read("orders", ["o_orderkey", "o_orderstatus"], CR(2) == Literal(70), table_defs)
        order = order.filter(CR(1) == Literal(70), [0])
        joined2 = order.broadcast_join(
            joined2,
            CR(0) == CR(2),
            [1, 2, 3],
            unique_keys_policy=UniqueKeysPolicy.left,
            perfect_hashing=True,
        )

        # joined3 has columns ["l_suppkey", "l_orderkey", "s_name"] which satisfies
        # exists (
        #     select * from
        #     lineitem l2
        #     where
        #         l2.l_orderkey = l1.l_orderkey
        #         and l2.l_suppkey <> l1.l_suppkey
        # )
        l3 = read("lineitem", ["l_suppkey", "l_orderkey"], None, table_defs)
        l3_probe = Q21LeftSemiJoinProbeRelation(joined2, l3)
        joined3 = Q21LeftSemiJoinRetrieveRelation(joined2, l3_probe)

        # group by
        #     s_name
        # order by
        #     numwait desc, s_name
        # limit 100
        result = (
            joined3.aggregate([CR(0)], [("count_all", CR(0))])
            .sort([(CR(1), "descending", "before"), (CR(0), "ascending", "before")])
            .fetch(0, 100)
        )

        return result
