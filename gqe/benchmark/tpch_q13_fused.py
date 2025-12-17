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
from gqe.benchmark.hardcoded.bindings.tpch_q13 import (
    Q13FusedFilterProbeRelation,
    Q13GroupjoinBuildRelation,
    Q13GroupjoinRetrieveRelation,
)
from gqe.benchmark.query import Query
from gqe.expression import ColumnReference as CR
from gqe.table_definition import TPCHTableDefinitions


class tpch_q13_fused(Query):
    """
    Creates a TPC-H Q13 phyiscal query plan with fused filter and probe operators.

    # Notes

    See `q13_opt` for more information on optimizations and limitations.
    """

    def __init__(self, scale_factor: float):
        """
        Parameters:
           scale_factor (float): The TPC-H scale factor.
        """

        self.scale_factor = scale_factor

    def root_relation(self, table_defs: TPCHTableDefinitions):
        # Read customer table
        customer = read("customer", ["c_custkey"], None, table_defs)

        # Read orders table and filter rows where o_comment does not contain '%special%requests%'
        orders = read("orders", ["o_custkey", "o_comment"], None, table_defs)

        # Build the groupjoin hash map
        customer_build = Q13GroupjoinBuildRelation(customer, self.scale_factor)

        # Probe the groupjoin hash map
        groupjoin_intermediate = Q13FusedFilterProbeRelation(customer_build, orders)

        # Group by c_custkey and calculate count(o_orderkey) as c_count
        # After aggregation, the result contains: c_custkey, c_count
        grouped_customer_orders = Q13GroupjoinRetrieveRelation(groupjoin_intermediate)

        # Group by c_count and calculate count(*) as custdist
        # After aggregation, the result contains: c_count, custdist
        grouped_c_count = grouped_customer_orders.aggregate([CR(1)], [("count_all", CR(1))])

        # Sort by custdist DESC, c_count DESC
        result = grouped_c_count.sort(
            [(CR(1), "descending", "before"), (CR(0), "descending", "before")]
        )

        return result
