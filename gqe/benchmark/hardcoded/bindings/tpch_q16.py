# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe.relation import Relation
import gqe.lib


class Q16FusedFilterJoinRelation(Relation):
    """
    Q16 specific fused filter and join relation.
    """

    def __init__(
        self, supplier_table: Relation, part_table: Relation, partsupp_table: Relation
    ):
        self.supplier_table = supplier_table
        self.part_table = part_table
        self.partsupp_table = partsupp_table

    def _to_cpp(self):
        return gqe.lib.q16_fused_filter_join(
            self.supplier_table._cpp,
            self.part_table._cpp,
            self.partsupp_table._cpp,
        )


class Q16AggregationRelation(Relation):
    """
    Q16 specific aggregation relation to count distinct suppliers.
    """

    def __init__(self, input: Relation):
        self.input = input

    def _to_cpp(self):
        return gqe.lib.q16_aggregate(self.input._cpp)
