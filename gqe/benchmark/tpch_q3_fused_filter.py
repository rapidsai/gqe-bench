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


class tpch_q3_fused_filter(Query):
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

        # Filter orders table: o_orderdate < date '1995-03-15'
        orders_filter = CR(2) < DateLiteral("1995-03-15")
        # Filter customer table: c_mktsegment = 'BUILDING'
        customer_filter = CR(5) == Literal("BUILDING")

        # Join customer and orders tables: c_custkey = o_custkey
        # orders_customer: o_orderkey, o_orderdate, o_shippriority
        orders_customer = orders.broadcast_join(
            customer,
            CR(1) == CR(4),  # o_custkey = c_custkey
            [0, 2, 3],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
            left_filter=orders_filter,
            right_filter=customer_filter,
        )

        # Filter lineitem table: l_shipdate > date '1995-03-15'
        lineitem_filter = CR(3) > DateLiteral("1995-03-15")

        # Join orders and lineitem tables: o_orderkey = l_orderkey
        # After join indices: l_orderkey, o_orderdate, o_shippriority, l_extendedprice, l_discount
        joined = lineitem.broadcast_join(
            orders_customer,
            CR(0) == CR(4),
            [0, 5, 6, 1, 2],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
            left_filter=lineitem_filter,
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
