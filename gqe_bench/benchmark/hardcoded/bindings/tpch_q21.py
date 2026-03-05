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

import gqe_bench.lib
from gqe_bench.relation import Relation


class Q21LeftAntiJoinProbeRelation(Relation):
    """
    Q21 specific left anti join build and probe relation.
    """

    def __init__(self, left_table: Relation, right_table: Relation):
        self.left_table = left_table
        self.right_table = right_table

    def _to_cpp(self):
        return gqe_bench.lib.q21_left_anti_join(self.left_table._cpp, self.right_table._cpp)


class Q21LeftAntiJoinRetrieveRelation(Relation):
    """
    Q21 specific left anti join retrieve relation.
    """

    def __init__(self, left_table: Relation, probe: Relation):
        self.left_table = left_table
        self.probe = probe

    def _to_cpp(self):
        return gqe_bench.lib.q21_left_anti_join_retrieve(self.left_table._cpp, self.probe._cpp)


class Q21LeftSemiJoinProbeRelation(Relation):
    """
    Q21 specific left semi join build and probe relation.
    """

    def __init__(self, left_table: Relation, right_table: Relation):
        self.left_table = left_table
        self.right_table = right_table

    def _to_cpp(self):
        return gqe_bench.lib.q21_left_semi_join(self.left_table._cpp, self.right_table._cpp)


class Q21LeftSemiJoinRetrieveRelation(Relation):
    """
    Q21 specific left semi join retrieve relation.
    """

    def __init__(self, left_table: Relation, probe: Relation):
        self.left_table = left_table
        self.probe = probe

    def _to_cpp(self):
        return gqe_bench.lib.q21_left_semi_join_retrieve(self.left_table._cpp, self.probe._cpp)
