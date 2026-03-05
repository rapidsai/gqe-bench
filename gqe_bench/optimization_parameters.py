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

"""
Factory functions for creating OptimizationParameters objects.
"""

from __future__ import annotations

import gqe_bench.lib as lib


def parse_compression_format(format_str: str) -> lib.CompressionFormat:
    """Convert compression format string to enum."""
    format_map = {
        "none": lib.CompressionFormat.none,
        "ans": lib.CompressionFormat.ans,
        "lz4": lib.CompressionFormat.lz4,
        "snappy": lib.CompressionFormat.snappy,
        "gdeflate": lib.CompressionFormat.gdeflate,
        "deflate": lib.CompressionFormat.deflate,
        "cascaded": lib.CompressionFormat.cascaded,
        "zstd": lib.CompressionFormat.zstd,
        "gzip": lib.CompressionFormat.gzip,
        "bitcomp": lib.CompressionFormat.bitcomp,
        "best_compression_ratio": lib.CompressionFormat.best_compression_ratio,
        "best_decompression_speed": lib.CompressionFormat.best_decompression_speed,
    }
    return format_map.get(format_str, lib.CompressionFormat.none)


def from_query_context(parameter, data) -> lib.OptimizationParameters:
    """
    Build OptimizationParameters from query execution context and data info objects.

    :param parameter: Object with execution parameters (num_workers, num_partitions, etc.)
    :param data: Object with data parameters (char_type, compression_format, etc.)
    :return: Configured OptimizationParameters instance.
    """
    params = lib.OptimizationParameters()
    params.max_num_workers = parameter.num_workers
    params.max_num_partitions = parameter.num_partitions
    params.use_opt_type_for_single_char_col = data.char_type == "char"
    params.use_overlap_mtx = parameter.use_overlap_mtx
    params.join_use_hash_map_cache = parameter.join_use_hash_map_cache
    params.read_zero_copy_enable = parameter.read_use_zero_copy
    params.join_use_unique_keys = parameter.join_use_unique_keys
    params.join_use_perfect_hash = parameter.join_use_perfect_hash
    params.join_use_mark_join = parameter.join_use_mark_join
    params.in_memory_table_compression_format = parse_compression_format(data.compression_format)
    params.in_memory_table_compression_chunk_size = data.compression_chunk_size
    params.use_partition_pruning = parameter.use_partition_pruning
    params.zone_map_partition_size = data.zone_map_partition_size
    params.filter_use_like_shift_and = parameter.filter_use_like_shift_and
    params.aggregation_use_perfect_hash = parameter.aggregation_use_perfect_hash
    return params


def from_catalog_context(cat_ctx) -> lib.OptimizationParameters:
    """
    Build initial OptimizationParameters from catalog context.

    Uses default values for execution-specific parameters that will be set
    when refresh_query_context is called.

    :param cat_ctx: Object with catalog parameters (use_opt_char_type, compression settings, etc.)
    :return: Configured OptimizationParameters instance.
    """
    params = lib.OptimizationParameters()
    params.max_num_workers = 1
    params.max_num_partitions = cat_ctx.num_row_groups
    params.use_opt_type_for_single_char_col = cat_ctx.use_opt_char_type
    params.in_memory_table_compression_format = parse_compression_format(
        cat_ctx.in_memory_table_compression_format
    )
    params.in_memory_table_compression_chunk_size = cat_ctx.in_memory_table_compression_chunk_size
    params.zone_map_partition_size = cat_ctx.zone_map_partition_size
    params.in_memory_table_compression_ratio_threshold = (
        cat_ctx.in_memory_table_compression_ratio_threshold
    )
    params.in_memory_table_secondary_compression_format = parse_compression_format(
        cat_ctx.in_memory_table_secondary_compression_format
    )
    params.in_memory_table_secondary_compression_ratio_threshold = (
        cat_ctx.in_memory_table_secondary_compression_ratio_threshold
    )
    params.in_memory_table_secondary_compression_multiplier_threshold = (
        cat_ctx.in_memory_table_secondary_compression_multiplier_threshold
    )
    params.in_memory_table_use_cpu_compression = cat_ctx.in_memory_table_use_cpu_compression
    params.in_memory_table_compression_level = cat_ctx.in_memory_table_compression_level
    return params
