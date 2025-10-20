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
from gqe.expression import ColumnReference as CR, LikeExpr, Literal
from gqe.benchmark.query import Query
from gqe.table_definition import TPCHTableDefinitions


"""
select
        c_count,
        count(*) as custdist
from
        (
                select
                        c_custkey,
                        count(o_orderkey) as c_count
                from
                        customer left outer join orders on
                                c_custkey = o_custkey
                                and o_comment not like '%special%requests%'
                group by
                        c_custkey
        ) as c_orders
group by
        c_count
order by
        custdist desc,
        c_count desc
"""


class tpch_q13(Query):
    """
    Creates a TPC-H Q13 phyiscal query plan.
    """

    def root_relation(self, table_defs: TPCHTableDefinitions):
        # Read customer table
        customer = read("customer", ["c_custkey"], None, table_defs)

        # Read orders table and filter rows where o_comment does not contain '%special%requests%'
        orders = read("orders", ["o_custkey", "o_comment"], None, table_defs).filter(
            LikeExpr(CR(1), "%special%requests%") == Literal(False), [0]
        )

        # Optimize I/O by skipping o_orderkey. Instead, generate a dummy
        # o_orderkey column for the left outer join.
        orders = orders.project([Literal(1), CR(0)])

        # Perform a left outer join between customer and orders on c_custkey = o_custkey
        # After the join, the result contains: c_custkey, o_orderkey
        customer_orders = customer.broadcast_join(
            orders, CR(0) == CR(2), [0, 1], "left"
        )

        # Group by c_custkey and calculate count(o_orderkey) as c_count
        # After aggregation, the result contains: c_custkey, c_count
        grouped_customer_orders = customer_orders.aggregate(
            [CR(0)], [("count_valid", CR(1))]
        )

        # Group by c_count and calculate count(*) as custdist
        # After aggregation, the result contains: c_count, custdist
        grouped_c_count = grouped_customer_orders.aggregate(
            [CR(1)], [("count_all", CR(1))]
        )

        # Sort by custdist DESC, c_count DESC
        result = grouped_c_count.sort(
            [(CR(1), "descending", "before"), (CR(0), "descending", "before")]
        )

        return result
