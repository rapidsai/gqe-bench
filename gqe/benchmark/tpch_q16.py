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
from gqe.expression import LikeExpr, Literal
from gqe.lib import UniqueKeysPolicy
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


class tpch_q16(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # Filter suppliers whose comments contain 'Customer Complaints'
        suppliers_with_complaints = read(
            "supplier", ["s_suppkey", "s_comment"], None, table_defs
        ).filter(LikeExpr(CR(1), "%Customer%Complaints%"), [0])

        # Read partsupp table and exclude suppliers with complaints
        partsupp = read("partsupp", ["ps_partkey", "ps_suppkey"], None, table_defs)

        # Filter parts that meet the conditions, selective filters are put first
        part = read("part", ["p_partkey", "p_brand", "p_type", "p_size"], None, table_defs).filter(
            (
                (CR(3) == Literal(49))
                | (CR(3) == Literal(14))
                | (CR(3) == Literal(23))
                | (CR(3) == Literal(45))
                | (CR(3) == Literal(19))
                | (CR(3) == Literal(3))
                | (CR(3) == Literal(36))
                | (CR(3) == Literal(9))
            )
            & (CR(1) != Literal("Brand#45"))
            & (LikeExpr(CR(2), "MEDIUM POLISHED%") == Literal(False)),
            [0, 1, 2, 3],
        )

        # Join part and partsupp
        # part: p_partkey(0), p_brand(1), p_type(2), p_size(3)
        # partsupp: ps_partkey(0), ps_suppkey(1)
        # joined: p_brand(0), p_type(1), p_size(2), ps_suppkey(3)
        joined = part.broadcast_join(
            partsupp,
            CR(0) == CR(4),
            [1, 2, 3, 5],
            unique_keys_policy=UniqueKeysPolicy.left,
            perfect_hashing=True,
        )

        # Exclude suppliers with complaints
        joined = joined.broadcast_join(
            suppliers_with_complaints, CR(3) == CR(4), [0, 1, 2, 3], "left_anti"
        )

        # Group by p_brand, p_type, p_size and count distinct ps_suppkey
        # Note: count(distinct ps_suppkey) is not directly supported in GQE,
        # but for TPC-H datasets, we can achieve this by grouping first and then counting.
        # Here, we first group by p_brand, p_type, p_size, ps_suppkey, and then group by p_brand, p_type, p_size to count.
        distinct_suppliers = joined.aggregate(
            [
                CR(0),
                CR(1),
                CR(2),
                CR(3),
            ],  # Group by p_brand, p_type, p_size, ps_suppkey
            [],  # No aggregation needed
        )

        # Group by p_brand, p_type, p_size and count
        result = distinct_suppliers.aggregate(
            [CR(0), CR(1), CR(2)],  # Group by p_brand, p_type, p_size
            [("count_all", CR(0))],  # count(*) as supplier_cnt
        ).sort(
            [
                (CR(3), "descending", "before"),  # Sort by supplier_cnt descending
                (CR(0), "ascending", "before"),  # Sort by p_brand ascending
                (CR(1), "ascending", "before"),  # Sort by p_type ascending
                (CR(2), "ascending", "before"),  # Sort by p_size ascending
            ]
        )

        return result
