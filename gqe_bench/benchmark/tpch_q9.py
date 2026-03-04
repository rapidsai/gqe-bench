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
from gqe_bench.expression import DatePartExpr, LikeExpr, Literal
from gqe_bench.lib import UniqueKeysPolicy
from gqe_bench.table_definition import TPCHTableDefinitions

"""
-- TPC-H Query 9
select
        nation,
        o_year,
        sum(amount) as sum_profit
from
        (
                select
                        n_name as nation,
                        extract(year from o_orderdate) as o_year,
                        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
                from
                        part,
                        supplier,
                        lineitem,
                        partsupp,
                        orders,
                        nation
                where
                        s_suppkey = l_suppkey
                        and ps_suppkey = l_suppkey
                        and ps_partkey = l_partkey
                        and p_partkey = l_partkey
                        and o_orderkey = l_orderkey
                        and s_nationkey = n_nationkey
                        and p_name like '%green%'
        ) as profit
group by
        nation,
        o_year
order by
        nation,
        o_year desc

List of optimization over substrait plans:
- Can correctly identify Join3 can use unique key join, currently the optimizer cannot handle compound unique keys.
"""


class tpch_q9(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        partsupp = read("partsupp", ["ps_suppkey", "ps_partkey", "ps_supplycost"], None, table_defs)

        # part has "p_partkey"
        part = read("part", ["p_partkey", "p_name"], None, table_defs).filter(
            LikeExpr(CR(1), "%green%"), [0]
        )

        # joining: "ps_suppkey", "ps_partkey", "ps_supplycost" with "p_partkey"
        # join1 has "ps_suppkey", "ps_supplycost", "p_partkey""
        join1 = partsupp.broadcast_join(
            part,
            CR(1) == CR(3),
            [0, 2, 3],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        supplier = read("supplier", ["s_suppkey", "s_nationkey"], None, table_defs)

        # Joining: "ps_suppkey", "ps_supplycost", "p_partkey"" with "s_suppkey", "s_nationkey"
        # Join2 has "ps_supplycost", "p_partkey", "s_suppkey", "s_nationkey"
        join2 = join1.broadcast_join(
            supplier,
            CR(0) == CR(3),
            [1, 2, 3, 4],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        lineitem = read(
            "lineitem",
            [
                "l_suppkey",
                "l_partkey",
                "l_orderkey",
                "l_extendedprice",
                "l_discount",
                "l_quantity",
            ],
            None,
            table_defs,
        )

        # Joining: "l_suppkey", "l_partkey", "l_orderkey", "l_extendedprice", "l_discount", "l_quantity" with
        # "ps_supplycost", "p_partkey", "s_suppkey", "s_nationkey"
        # "l_orderkey", "l_extendedprice", "l_discount", "l_quantity", "ps_supplycost", "s_nationkey"
        join3 = lineitem.broadcast_join(
            join2,
            (CR(0) == CR(8)) & (CR(1) == CR(7)),
            [2, 3, 4, 5, 6, 9],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # "l_orderkey", "amount", "s_nationkey"
        join3_projected = join3.project(
            [CR(0), CR(1) * Literal(1.0) - CR(1) * CR(2) - CR(4) * CR(3), CR(5)]
        )

        orders = read("orders", ["o_orderkey", "o_orderdate"], None, table_defs)

        # Joining "o_orderkey", "o_orderdate" with "l_orderkey", "amount", "s_nationkey"
        # "o_orderdate", "amount",  "s_nationkey"
        join4 = orders.broadcast_join(
            join3_projected,
            CR(0) == CR(2),
            [1, 3, 4],
            unique_keys_policy=UniqueKeysPolicy.left,
            perfect_hashing=True,
        )

        nation = read("nation", ["n_nationkey", "n_name"], None, table_defs)

        # Joining:  "o_orderdate", "amount",  "s_nationkey" with "n_nationkey", "n_name"
        # "n_name", "o_orderdate", "amount"
        join5 = join4.broadcast_join(
            nation,
            CR(2) == CR(3),
            [4, 0, 1],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        agg = join5.aggregate(
            [CR(0), DatePartExpr(CR(1), "year")],
            [("sum", CR(2))],
            perfect_hashing=False,
        )
        sorted_output = agg.sort([(CR(0), "ascending", "before"), (CR(1), "descending", "before")])

        return sorted_output
