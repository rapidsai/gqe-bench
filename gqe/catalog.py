# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class Catalog:
    def __init__(self) -> None:
        self._catalog = gqe.lib.Catalog()

    def register_tpch(
        self,
        dataset: str,
        storage: str = "parquet",
        num_row_groups: int = 8,
        load_data_of_query: int = 0,
        identifier_type: gqe.lib.TypeId = gqe.lib.TypeId.int32,
        use_opt_char_type: bool = True
    ) -> None:
        """
        Register TPC-H dataset in the catalog.

        :arg dataset: Location of the TPC-H dataset.
        :arg storage: Can be either `"parquet"` for loading from Parquet files during execution, or
            `"memory"` for pre-copying dataset into CPU memory during registration time.
        :arg num_row_groups: Number of row groups for in-memory storage.
        :arg load_data_of_query: For in-memory storage,
            if `load_data_of_query = 0` loads entire dataset,
            else if `0 < load_data_of_query <= 22` loads table and columns required for the
            specific query
        :arg identifier_type: Can be either `gqe.lib.TypeId.int32` or `gqe.lib.TypeId.int64`
        :arg use_opt_char_type: If true, use optimized char type for single character columns.
        
        """
        
        table_definitions = TPCHTableDefinitions(identifier_type, use_opt_char_type)
        
        if storage == "parquet":
            gqe.lib.register_tpch_parquet(self._catalog, dataset, table_definitions.query_table_definitions(0))
        elif storage == "memory":
            gqe.lib.register_tpch_in_memory(
                self._catalog, dataset, num_row_groups, table_definitions.query_table_definitions(load_data_of_query)
            )
        else:
            raise ValueError(f"Unrecognized storage: {storage}")

    def load_substrait(
        self, substrait_file: str, optimized: bool = True
    ) -> gqe.lib.Relation:
        return gqe.lib.load_substrait(self._catalog, substrait_file, optimized)
