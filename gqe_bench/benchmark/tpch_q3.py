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
select
  l_orderkey,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  o_orderdate,
  o_shippriority
from
  customer,
  orders,
  lineitem
where
  c_mktsegment = 'BUILDING'
  and c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate < date '1995-03-15'
  and l_shipdate > date '1995-03-15'
group by
  l_orderkey,
  o_orderdate,
  o_shippriority
order by
  revenue desc,
  o_orderdate
limit 10;


Optimization over substrait plan:
This identifies the o_orderkey in order_customer table is unique. Hence, the join of lineitem with order_customer is a unique key join.
The PK-FK join between order and customer on "o_custkey = c_custkey" doesnot change the uniqueness property of o_orderkey column.
"""


class tpch_q3(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        customer = read(
            "customer",
            ["c_custkey", "c_mktsegment"],
            (CR(6) == Literal("BUILDING")),
            table_defs,
        )
        orders = read(
            "orders",
            ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
            (CR(4) < DateLiteral("1995-03-15")),
            table_defs,
        )
        lineitem = read(
            "lineitem",
            ["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"],
            (CR(10) > DateLiteral("1995-03-15")),
            table_defs,
        )

        # Filter customer table: c_mktsegment = 'BUILDING'
        # filtered_customer: c_custkey(0)
        filtered_customer = customer.filter(CR(1) == Literal("BUILDING"), [0])

        # Filter orders table: o_orderdate < date '1995-03-15'
        # filtered_orders: o_orderkey, o_orderdate, o_shippriority
        filtered_orders = orders.filter(CR(2) < DateLiteral("1995-03-15"), [0, 1, 2, 3])

        # Join customer and orders tables: c_custkey = o_custkey
        # orders_customer: o_orderkey, o_orderdate, o_shippriority
        orders_customer = filtered_orders.broadcast_join(
            filtered_customer,
            CR(1) == CR(4),  # o_custkey = c_custkey
            [0, 2, 3],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # Filter lineitem table: l_shipdate > date '1995-03-15'
        # filtered_lineitem: l_orderkey, l_extendedprice, l_discount
        filtered_lineitem = lineitem.filter(
            CR(3) > DateLiteral("1995-03-15"),
            [0, 1, 2],
        )

        # Join orders and lineitem tables: o_orderkey = l_orderkey
        # After join indices: l_orderkey, o_orderdate, o_shippriority, l_extendedprice, l_discount
        joined = filtered_lineitem.broadcast_join(
            orders_customer,
            CR(0) == CR(3),
            [0, 4, 5, 1, 2],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # Group by: l_orderkey, o_orderdate, o_shippriority
        # Aggregate function: sum(l_extendedprice * (1 - l_discount)) as revenue
        # After aggregation: l_orderkey, o_orderdate, o_shippriority, revenue
        aggregated = joined.aggregate(
            [
                CR(0),
                CR(1),
                CR(2),
            ],
            [
                (
                    "sum",
                    CR(3) * (Literal(1) - CR(4)),
                )
            ],
            perfect_hashing=True,
        )

        # Sort by: revenue DESC, o_orderdate ASC
        sorted_result = aggregated.sort(
            [
                (CR(3), "descending", "after"),  # revenue DESC
                (CR(1), "ascending", "after"),  # o_orderdate ASC
            ]
        )

        # Fetch top 10 rows
        fetched = sorted_result.fetch(0, 10)

        # Project to match SQL output column order: l_orderkey, revenue, o_orderdate, o_shippriority
        result = fetched.project([CR(0), CR(3), CR(1), CR(2)])

        return result
