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


class Q21LeftAntiJoinProbeRelation(Relation):
    """
    Q21 specific left anti join build and probe relation.
    """

    def __init__(self, left_table: Relation, right_table: Relation):
        self.left_table = left_table
        self.right_table = right_table

    def _to_cpp(self):
        return gqe.lib.q21_left_anti_join(self.left_table._cpp, self.right_table._cpp)


class Q21LeftAntiJoinRetrieveRelation(Relation):
    """
    Q21 specific left anti join retrieve relation.
    """

    def __init__(self, left_table: Relation, probe: Relation):
        self.left_table = left_table
        self.probe = probe

    def _to_cpp(self):
        return gqe.lib.q21_left_anti_join_retrieve(
            self.left_table._cpp, self.probe._cpp
        )


class Q21LeftSemiJoinProbeRelation(Relation):
    """
    Q21 specific left semi join build and probe relation.
    """

    def __init__(self, left_table: Relation, right_table: Relation):
        self.left_table = left_table
        self.right_table = right_table

    def _to_cpp(self):
        return gqe.lib.q21_left_semi_join(self.left_table._cpp, self.right_table._cpp)


class Q21LeftSemiJoinRetrieveRelation(Relation):
    """
    Q21 specific left semi join retrieve relation.
    """

    def __init__(self, left_table: Relation, probe: Relation):
        self.left_table = left_table
        self.probe = probe

    def _to_cpp(self):
        return gqe.lib.q21_left_semi_join_retrieve(
            self.left_table._cpp, self.probe._cpp
        )
