# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe.relation import Relation
from gqe import Catalog
import gqe.lib


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
        in_memory_table_compression_format: str = "none",
        in_memory_table_compression_data_type: str = "char",
        compression_chunk_size: int = 65536,
        debug_mem_usage=False,
        use_partition_pruning: bool = False,
        zone_map_partition_size: int = 100000,
    ):
        self._context = gqe.lib.Context(
            max_num_workers,
            max_num_partitions,
            in_memory_table_compression_format,
            in_memory_table_compression_data_type,
            compression_chunk_size,
            use_opt_type_for_single_char_col,
            use_overlap_mtx,
            join_use_hash_map_cache,
            read_use_zero_copy,
            join_use_unique_keys,
            join_use_perfect_hash,
            debug_mem_usage,
            use_partition_pruning,
            zone_map_partition_size,
        )

    def execute(
        self,
        catalog: Catalog,
        relation: Relation | gqe.lib.Relation,
        output_path: str | None,
    ) -> float:
        """
        Execute the query plan.

        :param catalog: Catalog to execute the query plan on.
        :param relation: Root relation for the query plan.
        :param output_path: Path to write the output of `relation` to a Parquet file if this
            argument is valid `str`. If this argument is `None`, the output is not written. Note
            that the behavior is undefined if `output_path` is valid but `relation` does not produce
            an output.

        :return: The execution time in ms.
        """
        if isinstance(relation, Relation):
            relation = relation._to_cpp()

        return self._context.execute(catalog._catalog, relation, output_path)
