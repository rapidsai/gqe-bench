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
from gqe.expression import ColumnReference as CR, LikeExpr
from gqe.benchmark.query import Query
from gqe.relation import Relation
from gqe.benchmark.hardcoded.bindings.tpch_q16 import (
    Q16FusedFilterJoinRelation,
    Q16AggregationRelation,
)
from gqe.table_definition import TPCHTableDefinitions


"""
select
        p_brand,
        p_type,
        p_size,
        count(distinct ps_suppkey) as supplier_cnt
from
        partsupp,
        part
where
        p_partkey = ps_partkey
        and p_brand <> 'Brand#45'
        and p_type not like 'MEDIUM POLISHED%'
        and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
        and ps_suppkey not in (
                select
                        s_suppkey
                from
                        supplier
                where
                        s_comment like '%Customer%Complaints%'
        )
group by
        p_brand,
        p_type,
        p_size
order by
        supplier_cnt desc,
        p_brand,
        p_type,
        p_size
"""


# Optimizations:
# - Fuse the filter on part table and hash table build on part table into a single operator.
# - Fuse the probing on partsupp table against part table and supplier table into a single operator.
# - Concatenate the aggregation partitions into one partition and only perform aggregation once.
class tpch_q16_opt(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # Filter suppliers whose comments contain 'Customer Complaints'
        # This filter is less time consuming, then leave it without optimization for now.
        suppliers_with_complaints = read(
            "supplier", ["s_suppkey", "s_comment"], None, table_defs
        ).filter(LikeExpr(CR(1), "%Customer%Complaints%"), [0])

        part = read(
            "part", ["p_partkey", "p_brand", "p_type", "p_size"], None, table_defs
        )

        # Read partsupp table and exclude suppliers with complaints
        partsupp = read("partsupp", ["ps_partkey", "ps_suppkey"], None, table_defs)

        joined = Q16FusedFilterJoinRelation(suppliers_with_complaints, part, partsupp)

        # Perform the aggregation of count(distinct ps_suppkey) and grouping by p_brand, p_type, p_size.
        # This is a custom aggregation by only doing one aggregation.
        # FIXME: if later aggregation optimization !MR320 is merged, we can replace this with standard aggregation.
        result = Q16AggregationRelation(joined)

        result = result.sort(
            [
                (CR(3), "descending", "before"),  # Sort by supplier_cnt descending
                (CR(0), "ascending", "before"),  # Sort by p_brand ascending
                (CR(1), "ascending", "before"),  # Sort by p_type ascending
                (CR(2), "ascending", "before"),  # Sort by p_size ascending
            ]
        )

        return result
