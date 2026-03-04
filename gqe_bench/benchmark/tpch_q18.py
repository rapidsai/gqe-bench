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
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice,
        sum(l_quantity)
from
        customer,
        orders,
        lineitem
where
        o_orderkey in (
                select
                        l_orderkey
                from
                        lineitem
                group by
                        l_orderkey having
                                sum(l_quantity) > 300
        )
        and c_custkey = o_custkey
        and o_orderkey = l_orderkey
group by
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice
order by
        o_totalprice desc,
        o_orderdate
limit
        100
"""


class tpch_q18(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # After this operation, `lineitem` contains [l_orderkey, sum(l_quantity)]
        lineitem = (
            read("lineitem", ["l_orderkey", "l_quantity"], None, table_defs)
            .aggregate([CR(0)], [("sum", CR(1))], perfect_hashing=True)
            .filter(CR(1) > Literal(300.0), [0, 1])
        )

        customer = read("customer", ["c_custkey", "c_name"], None, table_defs)

        # After this operation, `orders` contains
        # [o_orderkey, o_custkey, o_orderdate, o_totalprice, sum(l_quantity)]
        orders = read(
            "orders",
            ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"],
            None,
            table_defs,
        ).broadcast_join(
            lineitem,
            CR(0) == CR(4),
            [0, 1, 2, 3, 5],
            unique_keys_policy=UniqueKeysPolicy.left,
            perfect_hashing=True,
        )

        # After this operation, `orders` contains
        # [o_orderkey, c_custkey, o_orderdate, o_totalprice, sum(l_quantity), c_name]
        orders = orders.broadcast_join(
            customer,
            CR(1) == CR(5),
            [0, 1, 2, 3, 4, 6],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # After this operation, `orders` contains
        # [c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)]
        orders = (
            orders.aggregate(
                [CR(5), CR(1), CR(0), CR(2), CR(3)],
                [("sum", CR(4))],
                perfect_hashing=False,
            )
            .sort([(CR(4), "descending", "before"), (CR(3), "ascending", "before")])
            .fetch(0, 100)
        )

        return orders
