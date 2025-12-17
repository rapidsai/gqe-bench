# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import gqe.lib
from gqe import read
from gqe.benchmark.query import Query
from gqe.expression import ColumnReference as CR
from gqe.lib import UniqueKeysPolicy
from gqe.relation import Relation
from gqe.table_definition import TPCHTableDefinitions


class Q18GroupByRelation(Relation):
    """
    Q18 specific group by relation that performs aggregation on lineitem
    table with atomic initialization of buckets.
    """

    def __init__(self, lineitem: Relation, scale_factor: int):
        self.input = lineitem  # Required by fix_partial_filter_column_references
        self.scale_factor = scale_factor

    def _to_cpp(self):
        return gqe.lib.q18_groupby(self.input._cpp, int(self.scale_factor))


"""
TPC-H Q18 Optimized: Large Volume Customer Query with Group By Optimization

This is the optimized version of TPC-H Q18 that finds customers who have placed
orders with large quantities. It uses a specialized group by approach that first
aggregates lineitem quantities, then joins with orders.

The group by optimization provides better performance by:
1. Direct key comparison (l_orderkey) instead of device_row_comparator, eliminating indirection
2. Single-pass atomic aggregation using insert_or_apply that atomically initializes buckets and
   accumulates quantities, eliminating separate insert-then-find operations
3. Integrated filtering (sum_quantity > 300) during hash table retrieval, avoiding materialization
   of unfiltered results as a separate operation

Original Query:
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

Group By Approach:
1. Group by phase: Aggregate sum(l_quantity) by l_orderkey from lineitem table
2. Filter phase: Keep only orders with sum(l_quantity) > 300
3. Join with orders table to get order information
4. Join with customer table and final aggregation
"""


class tpch_q18_opt(Query):
    def __init__(self, scale_factor: int):
        """
        Parameters:
           scale_factor (int): The TPC-H scale factor.
        """
        self.scale_factor = scale_factor

    def root_relation(self, table_defs: TPCHTableDefinitions):
        # Get scale factor for sizing estimates
        scale_factor = self.scale_factor

        # Step 1: Group By on lineitem - Aggregate quantities by orderkey
        # Input: lineitem [l_orderkey, l_quantity]
        # Output: [l_orderkey, sum_l_quantity] where sum_l_quantity > 300
        lineitem = read("lineitem", ["l_orderkey", "l_quantity"])

        # Perform the group by operation with atomic initialization
        # This aggregates lineitem quantities and filters for > 300
        aggregated_lineitem = Q18GroupByRelation(lineitem, scale_factor)

        # Step 2: Join aggregated results with orders table to get full order information
        # aggregated_lineitem: [l_orderkey, sum_l_quantity]
        # orders: [o_orderkey, o_custkey, o_orderdate, o_totalprice]
        # Result: [l_orderkey, sum_l_quantity, o_custkey, o_orderdate, o_totalprice]
        orders = read("orders", ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"])
        orders_with_quantities = aggregated_lineitem.broadcast_join(
            orders,
            CR(0) == CR(2),  # l_orderkey == o_orderkey
            [
                0,
                1,
                3,
                4,
                5,
            ],  # [l_orderkey, sum_l_quantity, o_custkey, o_orderdate, o_totalprice]
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # Step 3: Join with customer table
        customer = read("customer", ["c_custkey", "c_name"])

        # Join orders with customer
        # orders_with_quantities: [l_orderkey, sum_l_quantity, o_custkey, o_orderdate, o_totalprice]
        # customer: [c_custkey, c_name]
        # Result: [l_orderkey, sum_l_quantity, o_custkey, o_orderdate, o_totalprice, c_name]
        orders_with_customer = orders_with_quantities.broadcast_join(
            customer,
            CR(2) == CR(5),  # o_custkey == c_custkey
            [
                0,
                1,
                2,
                3,
                4,
                6,
            ],  # [l_orderkey, sum_l_quantity, o_custkey, o_orderdate, o_totalprice, c_name]
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # Step 4: Final aggregation and sorting
        # Group by: [c_name, c_custkey, l_orderkey, o_orderdate, o_totalprice]
        # Aggregate: sum(l_quantity) - since Group By already aggregated, this is identity
        # orders_with_customer: [l_orderkey, sum_l_quantity, o_custkey, o_orderdate, o_totalprice, c_name]
        # Result: [c_name, c_custkey, l_orderkey, o_orderdate, o_totalprice, sum_l_quantity]
        result = (
            orders_with_customer.aggregate(
                [
                    CR(5),
                    CR(2),
                    CR(0),
                    CR(3),
                    CR(4),
                ],  # [c_name, o_custkey, l_orderkey, o_orderdate, o_totalprice]
                [("sum", CR(1))],  # sum(l_quantity) - already aggregated, so this is identity
            )
            .sort(
                [
                    (
                        CR(4),
                        "descending",
                        "before",
                    ),  # o_totalprice desc (now at index 4)
                    (CR(3), "ascending", "before"),  # o_orderdate asc (now at index 3)
                ]
            )
            .fetch(0, 100)
        )

        return result
