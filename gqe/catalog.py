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
In GQE, a **catalog** contains the metadata of a database. This includes the tables and their
storage locations. Therefore, before we load a table using :func:`read <gqe.relation.read>`, we
have to register it in the Catalog.

At the moment, we can only register TPC-H tables through
:meth:`register_tpch <gqe.catalog.Catalog.register_tpch>`. Registering custom tables will be added
in the future.
"""

import gqe.lib
from .table_definition import TPCHTableDefinitions
from .execute import MultiProcessRuntimeContext


class Catalog:
    def __init__(self) -> None:
        self._catalog = gqe.lib.Catalog()

    def register_tpch(
        self,
        dataset: str,
        storage_kind: str = "pinned_memory",
        num_row_groups: int = 8,
        load_data_of_query: int = 0,
        load_all_data_from: str = "required",
        identifier_type: gqe.lib.TypeId = gqe.lib.TypeId.int32,
        use_opt_char_type: bool = True,
        in_memory_table_compression_format="none",
        in_memory_table_compression_data_type="char",
        compression_chunk_size=2**16,
        zone_map_partition_size=100000,
        multiprocess_runtime_context: MultiProcessRuntimeContext = None,
        debug_mem_usage: bool = False,
    ) -> TPCHTableDefinitions:
        """
        Register TPC-H dataset in the catalog.

        :arg dataset: Location of the TPC-H dataset.
        :arg storage_kind: Storage kind for tables. Can be either `"pinned_memory"`,
            `"system_memory"`, `"numa_memory"`, `"device_memory"`, or `"managed_memory"` or
            `"parquet_file"`, or `"boost_shared_memory"`.
        :arg num_row_groups: Number of row groups for in-memory storage.
        :arg load_data_of_query: For in-memory storage,
            if `load_data_of_query = 0` loads entire dataset,
            else if `0 < load_data_of_query <= 22` loads table and columns required for the
            specific query
        :arg load_all_data_from: Whether to load all data from the TPC-Hschema (`"full"`) or only the data required by the 22 TPC-H queries (`"required"`).
            if `load_all_data_from = "required"` loads data required by the 22 TPC-H queries,
            else if `load_all_data_from = "full"` loads all data from the TPC-H schema
        :arg identifier_type: Can be either `gqe.lib.TypeId.int32` or `gqe.lib.TypeId.int64`
        :arg use_opt_char_type: If true, use optimized char type for single character columns.
        :arg in_memory_table_compression_format: Compression format for the in-memory table.
        :arg in_memory_table_compression_data_type: Determines how input data is viewed as for compression.
        :arg compression_chunk_size: Size of each chunk for nvcomp compression.
        :arg zone_map_partition_size: Number of rows per zone map partition.
        :arg multiprocess_runtime_context: Context reused for multiprocess tasks execution.
        """
        table_definitions = TPCHTableDefinitions(identifier_type, use_opt_char_type)
        if storage_kind == "parquet_file":
            gqe.lib.register_tpch_parquet(
                self._catalog,
                dataset,
                table_definitions.query_table_definitions(
                    load_data_of_query, load_all_data_from
                ),
            )
        elif storage_kind in [
            "pinned_memory",
            "system_memory",
            "numa_memory",
            "device_memory",
            "managed_memory",
            "numa_pinned_memory",
            "boost_shared_memory",
        ]:

            gqe.lib.register_tpch_in_memory(
                self._catalog,
                dataset,
                num_row_groups,
                in_memory_table_compression_format,
                in_memory_table_compression_data_type,
                compression_chunk_size,
                zone_map_partition_size,
                table_definitions.query_table_definitions(
                    load_data_of_query, load_all_data_from
                ),
                storage_kind,
                multiprocess_runtime_context,
                debug_mem_usage,
            )
        else:
            raise ValueError(f"Unrecognized storage kind: {storage_kind}")
        return table_definitions

    def load_substrait(
        self,
        substrait_file: str,
        optimized: bool = True,
        multiprocess_runtime_context: MultiProcessRuntimeContext = None,
    ) -> gqe.lib.Relation:
        return gqe.lib.load_substrait(
            self._catalog, substrait_file, optimized, multiprocess_runtime_context
        )
