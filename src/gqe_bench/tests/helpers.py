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

"""Shared test factory helpers for constructing DataInfo, Query, and QueryParams."""

from gqe_bench.query_source import QuerySource
from gqe_bench.schema import DataInfo, Query, QueryParams


def make_data_info(**overrides: object) -> DataInfo:
    """Create a DataInfo with sensible defaults, overriding specific fields."""
    defaults = dict(
        location="/x",
        scale_factor=100,
        num_row_groups=8,
        identifier_type="int64",
        storage_device_kind="numa_pinned_memory",
        decimal_type="double",
        compression_format="none",
        compression_ratio_threshold=1.0,
        compression_chunk_size=131072,
        compression_level=10,
        zone_map_partition_size=200000,
        secondary_compression_format="none",
        secondary_compression_ratio_threshold=2.5,
        secondary_compression_multiplier_threshold=1.5,
        use_cpu_compression=False,
    )
    defaults.update(overrides)
    return DataInfo(**defaults)


def make_query(**overrides: object) -> Query:
    """Create a Query with sensible defaults."""
    defaults = dict(
        name="Q1",
        source=QuerySource.SQL,
        reference_file=None,
        content=b"SELECT 1",
    )
    defaults.update(overrides)
    return Query(**defaults)


def make_query_params(**overrides: object) -> QueryParams:
    """Create a QueryParams (sweep variant) with sensible defaults."""
    defaults = dict(
        num_workers=1,
        num_partitions=4,
        use_overlap_mtx=False,
        join_use_hash_map_cache=True,
        read_use_zero_copy=False,
        join_use_unique_keys=True,
        join_use_perfect_hash=False,
        join_use_mark_join=False,
        use_partition_pruning=False,
        filter_use_like_shift_and=False,
        aggregation_use_perfect_hash=False,
    )
    defaults.update(overrides)
    return QueryParams(**defaults)
