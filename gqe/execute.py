# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations
from typing import TYPE_CHECKING

from gqe.relation import Relation
import gqe.lib

# Circular import of between execute and catalog
if TYPE_CHECKING:
    from gqe.catalog import Catalog


class Context:
    def __init__(
        self,
        max_num_workers: int = 1,
        max_num_partitions: int = 8,
        use_opt_type_for_single_char_col: bool = False,
        use_overlap_mtx: bool = False,
        join_use_hash_map_cache: bool = False,
        read_use_zero_copy: bool = False,
        join_use_unique_keys: bool = False,
        join_use_perfect_hash: bool = False,
        join_use_mark_join: bool = False,
        in_memory_table_compression_format: str = "none",
        in_memory_table_compression_data_type: str = "char",
        compression_chunk_size: int = 65536,
        use_partition_pruning: bool = False,
        zone_map_partition_size: int = 100000,
        filter_use_like_shift_and: bool = False,
        aggregation_use_perfect_hash: bool = False,
        debug_mem_usage=False,
        cupti_metrics: list[str] | None = None,
    ):
        """
        Create a new context.

        See the GQE documentation for a description of standard parameters.

        Additional parameters are:

        :param debug_mem_usage=False,
        :param cupti_metrics: The CUPTI range metrics to profile. If this argument is `None`, the profiler is completely disabled.
        """

        self._context = gqe.lib.Context(
            max_num_workers,
            max_num_partitions,
            in_memory_table_compression_format,
            in_memory_table_compression_data_type,
            compression_chunk_size,
            zone_map_partition_size,
            debug_mem_usage,
            use_opt_type_for_single_char_col,
            use_overlap_mtx,
            join_use_hash_map_cache,
            read_use_zero_copy,
            join_use_unique_keys,
            join_use_perfect_hash,
            join_use_mark_join,
            use_partition_pruning,
            filter_use_like_shift_and,
            aggregation_use_perfect_hash,
            cupti_metrics,
        )

    def execute(
        self,
        catalog: Catalog,
        relation: Relation | gqe.lib.Relation,
        output_path: str | None,
    ) -> tuple[float, dict]:
        """
        Execute the query plan.

        :param catalog: Catalog to execute the query plan on.
        :param relation: Root relation for the query plan.
        :param output_path: Path to write the output of `relation` to a Parquet file if this
            argument is valid `str`. If this argument is `None`, the output is not written. Note
            that the behavior is undefined if `output_path` is valid but `relation` does not produce
            an output.

        :return: A tuple containing the execution time in seconds and a dictionary of the specified CUPTI metrics.
        """
        if isinstance(relation, Relation):
            relation = relation._to_cpp()

        return self._context.execute(catalog._catalog, relation, output_path)


class MultiProcessRuntimeContext:
    def __init__(self, scheduler_type: gqe.lib.scheduler_type, storage_kind: str):
        self._context = gqe.lib.MultiProcessRuntimeContext(scheduler_type, storage_kind)

    def get(self):
        return self._context.get()

    def finalize(self):
        self._context.finalize()


class MultiProcessContext:
    def __init__(
        self,
        runtime_context: MultiProcessRuntimeContext,
        max_num_workers: int = 1,
        max_num_partitions: int = 8,
        use_opt_type_for_single_char_col: bool = False,
        use_overlap_mtx: bool = False,
        join_use_hash_map_cache: bool = False,
        read_use_zero_copy: bool = False,
        join_use_unique_keys: bool = False,
        join_use_perfect_hash: bool = False,
        join_use_mark_join: bool = False,
        in_memory_table_compression_format: str = "none",
        in_memory_table_compression_data_type: str = "char",
        compression_chunk_size: int = 65536,
        use_partition_pruning: bool = False,
        zone_map_partition_size: int = 100000,
        filter_use_like_shift_and: bool = False,
        aggregation_use_perfect_hash: bool = False,
        scheduler_type: gqe.lib.scheduler_type = gqe.lib.scheduler_type.ROUND_ROBIN,
    ):
        self._context = gqe.lib.MultiProcessContext(
            runtime_context,
            max_num_workers,
            max_num_partitions,
            in_memory_table_compression_format,
            in_memory_table_compression_data_type,
            compression_chunk_size,
            zone_map_partition_size,
            scheduler_type,
            use_opt_type_for_single_char_col,
            use_overlap_mtx,
            join_use_hash_map_cache,
            read_use_zero_copy,
            join_use_unique_keys,
            join_use_perfect_hash,
            join_use_mark_join,
            use_partition_pruning,
            filter_use_like_shift_and,
            aggregation_use_perfect_hash,
        )

    def execute(
        self,
        catalog: Catalog,
        relation: Relation | gqe.lib.Relation,
        output_path: str | None,
    ) -> tuple[float, dict]:
        """
        Execute the query plan.

        :param catalog: Catalog to execute the query plan on.
        :param relation: Root relation for the query plan.
        :param output_path: Path to write the output of `relation` to a Parquet file if this
            argument is valid `str`. If this argument is `None`, the output is not written. Note
            that the behavior is undefined if `output_path` is valid but `relation` does not produce
            an output.

        :return: A tuple containing the execution time in seconds and a dictionary of the specified CUPTI metrics.
        """
        if isinstance(relation, Relation):
            relation = relation._to_cpp()

        return self._context.execute(catalog._catalog, relation, output_path)
