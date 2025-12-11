# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Factory functions for creating OptimizationParameters objects.
"""

from __future__ import annotations

import gqe.lib as lib


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


def parse_compression_data_type(data_type_str: str) -> lib.NvcompType:
    """Convert compression data type string to enum."""
    type_map = {
        "char": lib.NvcompType.char,
        "uchar": lib.NvcompType.uchar,
        "short": lib.NvcompType.short,
        "ushort": lib.NvcompType.ushort,
        "int": lib.NvcompType.int,
        "uint": lib.NvcompType.uint,
        "longlong": lib.NvcompType.longlong,
        "ulonglong": lib.NvcompType.ulonglong,
        "bits": lib.NvcompType.bits,
    }
    return type_map.get(data_type_str, lib.NvcompType.char)


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
    params.in_memory_table_compression_format = parse_compression_format(
        data.compression_format
    )
    params.in_memory_table_compression_data_type = parse_compression_data_type(
        data.compression_data_type
    )
    params.compression_chunk_size = data.compression_chunk_size
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
    params.in_memory_table_compression_data_type = parse_compression_data_type(
        cat_ctx.in_memory_table_compression_data_type
    )
    params.compression_chunk_size = cat_ctx.compression_chunk_size
    params.zone_map_partition_size = cat_ctx.zone_map_partition_size
    return params
