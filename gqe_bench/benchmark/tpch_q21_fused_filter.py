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

from gqe_bench import read
from gqe_bench.benchmark.query import Query
from gqe_bench.expression import ColumnReference as CR
from gqe_bench.expression import Literal
from gqe_bench.lib import UniqueKeysPolicy
from gqe_bench.table_definition import TPCHTableDefinitions

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


class tpch_q21_fused_filter(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # `supplier` has columns ["s_suppkey", "s_name"] which satisfies
        # s_nationkey = n_nationkey and n_name = 'SAUDI ARABIA'
        nation = read(
            "nation",
            ["n_nationkey", "n_name"],
            CR(1) == Literal("SAUDI ARABIA"),
            table_defs,
        )
        nation_filter = CR(4) == Literal("SAUDI ARABIA")
        supplier = read(
            "supplier", ["s_suppkey", "s_name", "s_nationkey"], None, table_defs
        ).broadcast_join(
            nation,
            CR(2) == CR(3),
            [0, 1],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
            right_filter=nation_filter,
        )

        # l1.l_receiptdate > l1.l_commitdate
        # `l1` has columns ["l_suppkey", "l_orderkey"]
        l1 = read(
            "lineitem",
            ["l_suppkey", "l_orderkey", "l_receiptdate", "l_commitdate"],
            CR(12) > CR(11),
            table_defs,
        )
        l1_filter = CR(2) > CR(3)

        # s_suppkey = l1.l_suppkey
        # `l1` has columns ["l_suppkey", "l_orderkey", "s_name"]
        l1 = l1.broadcast_join(
            supplier,
            CR(0) == CR(4),
            [0, 1, 5],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
            left_filter=l1_filter,
        )

        # l3.l_receiptdate > l3.l_commitdate
        # `l3` has columns ["l_suppkey", "l_orderkey"]
        l3 = read(
            "lineitem",
            ["l_suppkey", "l_orderkey", "l_receiptdate", "l_commitdate"],
            CR(12) > CR(11),
            table_defs,
        )
        # There is offset added based on the number of columns in l1
        l3_filter = CR(5) > CR(6)

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
            l3,
            (CR(1) == CR(4)) & (CR(0) != CR(3)),
            [0, 1, 2],
            "left_anti",
            True,
            right_filter=l3_filter,
        )

        # o_orderkey = l1.l_orderkey and o_orderstatus = 'F'
        order = read("orders", ["o_orderkey", "o_orderstatus"], CR(2) == Literal(70), table_defs)
        # o_orderstatus = 'F'
        order_filter = CR(1) == Literal(70)
        l1 = order.broadcast_join(
            l1,
            CR(0) == CR(3),
            [2, 3, 4],
            broadcast_left=True,  # This is fast compared to broadcast_right.
            unique_keys_policy=UniqueKeysPolicy.left,
            perfect_hashing=True,
            left_filter=order_filter,
        )

        # l1 has columns ["l_suppkey", "l_orderkey", "s_name"] which satisfies
        # exists (
        #     select * from
        #     lineitem l2
        #     where
        #         l2.l_orderkey = l1.l_orderkey
        #         and l2.l_suppkey <> l1.l_suppkey
        # )
        l2 = read("lineitem", ["l_suppkey", "l_orderkey"], None, table_defs)
        l1 = l1.broadcast_join(l2, (CR(1) == CR(4)) & (CR(0) != CR(3)), [2], "left_semi", True)

        # group by
        #     s_name
        # order by
        #     numwait desc, s_name
        # limit 100
        l1 = (
            l1.aggregate([CR(0)], [("count_all", CR(0))], perfect_hashing=False)
            .sort([(CR(1), "descending", "before"), (CR(0), "ascending", "before")])
            .fetch(0, 100)
        )

        return l1
