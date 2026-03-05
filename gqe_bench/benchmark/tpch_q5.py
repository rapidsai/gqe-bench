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
from gqe_bench.expression import DateLiteral, Literal
from gqe_bench.lib import UniqueKeysPolicy
from gqe_bench.table_definition import TPCHTableDefinitions

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
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # read customer table
        customer = read("customer", ["c_nationkey", "c_custkey"], None, table_defs)

        # read orders table
        orders = read(
            "orders",
            ["o_custkey", "o_orderkey", "o_orderdate"],
            (CR(4) >= DateLiteral("1994-01-01")) & (CR(4) <= DateLiteral("1994-12-31")),
            table_defs,
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
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )
        # result has ["o_orderkey", "c_nationkey"]

        # broadcast join - lineitem result is smaller
        lineitem = read(
            "lineitem",
            ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"],
            None,
            table_defs,
        )
        result = lineitem.broadcast_join(
            result,
            CR(0) == CR(4),
            [1, 2, 3, 5],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )
        # result has ["l_suppkey", "l_extendedprice", "l_discount", "c_nationkey"]

        # broadcast join - supplier is smaller
        supplier = read("supplier", ["s_suppkey", "s_nationkey"], None, table_defs)
        result = result.broadcast_join(
            supplier,
            (CR(0) == CR(4)) & (CR(3) == CR(5)),
            [1, 2, 5],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )
        # result has ["l_extendedprice","l_discount", "s_nationkey"]

        # broadcast join nation
        nation = read("nation", ["n_nationkey", "n_regionkey", "n_name"], None, table_defs)
        result = result.broadcast_join(
            nation,
            (CR(2) == CR(3)),
            [0, 1, 4, 5],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )
        # result has [ "l_extendedprice", "l_discount", "n_regionkey", "n_name"]

        # broadcast join -  after filter - region
        region = read("region", ["r_regionkey", "r_name"], None, table_defs).filter(
            CR(1) == Literal("ASIA"), [0]
        )
        result = result.broadcast_join(
            region,
            (CR(2) == CR(4)),
            [0, 1, 3],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )
        # result has [ "l_extendedprice", "l_discount",  "n_name"]

        # groupby and sort
        result = result.aggregate(
            [CR(2)], [("sum", CR(0) * (Literal(1) - CR(1)))], perfect_hashing=False
        )
        return result.sort([(CR(1), "descending", "before")])
