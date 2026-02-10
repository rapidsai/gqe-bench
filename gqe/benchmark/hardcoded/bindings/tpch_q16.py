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

import gqe.lib
from gqe.relation import Relation


class Q16FusedFilterJoinRelation(Relation):
    """
    Q16 specific fused filter and join relation.
    """

    def __init__(self, supplier_table: Relation, part_table: Relation, partsupp_table: Relation):
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
