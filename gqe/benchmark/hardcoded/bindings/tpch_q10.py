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


class Q10UniqueKeyInnerJoinBuildRelation(Relation):
    """
    Q10-specific unique-key inner join build relation.
    """

    def __init__(self, build_side_table: Relation, key_column_idx: int, enable_bloom_filter: bool):
        self.build_side_table = build_side_table
        self.key_column_idx = key_column_idx
        self.enable_bloom_filter = enable_bloom_filter

    def _to_cpp(self):
        return gqe.lib.q10_unique_key_inner_join_build(
            self.build_side_table._cpp, self.key_column_idx, self.enable_bloom_filter
        )


class Q10UniqueKeyInnerJoinProbeRelation(Relation):
    """
    Q10-specific unique-key inner join probe relation.
    """

    def __init__(
        self,
        build_side_map: Relation,
        build_side_table: Relation,
        probe_side_table: Relation,
        build_side_key_column_idx: int,
        probe_side_key_column_idx: int,
        projection_indices: list[int],
    ):
        self.build_side_map = build_side_map
        self.build_side_table = build_side_table
        self.probe_side_table = probe_side_table
        self.build_side_key_column_idx = build_side_key_column_idx
        self.probe_side_key_column_idx = probe_side_key_column_idx
        self.projection_indices = projection_indices

    def _to_cpp(self):
        return gqe.lib.q10_unique_key_inner_join_probe(
            self.build_side_map._cpp,
            self.build_side_table._cpp,
            self.probe_side_table._cpp,
            self.build_side_key_column_idx,
            self.probe_side_key_column_idx,
            self.projection_indices,
        )


class Q10FusedProbesJoinMultimapBuildRelation(Relation):
    """
    Q10-specific fused probes join multimap build relation.
    """

    def __init__(self, build_side_table: Relation, key_column_idx: int):
        self.build_side_table = build_side_table
        self.key_column_idx = key_column_idx

    def _to_cpp(self):
        return gqe.lib.q10_fused_probes_join_multimap_build(
            self.build_side_table._cpp, self.key_column_idx
        )


class Q10FusedProbesJoinMapBuildRelation(Relation):
    """
    Q10-specific fused probes join map build relation.
    """

    def __init__(self, build_side_table: Relation, key_column_idx: int):
        self.build_side_table = build_side_table
        self.key_column_idx = key_column_idx

    def _to_cpp(self):
        return gqe.lib.q10_fused_probes_join_map_build(
            self.build_side_table._cpp, self.key_column_idx
        )


class Q10FusedProbesJoinProbeRelation(Relation):
    """
    Q10-specific fused probes join probe relation.
    """

    def __init__(
        self,
        o_custkey_to_row_indices_multimap: Relation,
        n_nationkey_to_row_index_map: Relation,
        join_orders_lineitem_table: Relation,
        nation_table: int,
        customer_table: int,
    ):
        self.o_custkey_to_row_indices_multimap = o_custkey_to_row_indices_multimap
        self.n_nationkey_to_row_index_map = n_nationkey_to_row_index_map
        self.join_orders_lineitem_table = join_orders_lineitem_table
        self.nation_table = nation_table
        self.customer_table = customer_table

    def _to_cpp(self):
        return gqe.lib.q10_fused_probes_join_probe(
            self.o_custkey_to_row_indices_multimap._cpp,
            self.n_nationkey_to_row_index_map._cpp,
            self.join_orders_lineitem_table._cpp,
            self.nation_table._cpp,
            self.customer_table._cpp,
        )


class Q10SortLimitRelation(Relation):
    """
    Q10-specific sort-limit relation.
    """

    def __init__(
        self,
        input_table: Relation,
        key_column_idx: int,
        limit: int,
        projection_indices: list[int],
    ):
        self.input_table = input_table
        self.key_column_idx = key_column_idx
        self.limit = limit
        self.projection_indices = projection_indices

    def _to_cpp(self):
        return gqe.lib.q10_sort_limit(
            self.input_table._cpp,
            self.key_column_idx,
            self.limit,
            self.projection_indices,
        )
