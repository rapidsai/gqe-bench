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
In GQE, a **catalog** contains the metadata of a database. This includes the tables and their
storage locations. Therefore, before we load a table using :func:`read <gqe.relation.read>`, we
have to register it in the Catalog.

We can register both TPC-H tables and custom tables through
:meth:`register_tables <gqe.catalog.Catalog.register_tables>`.
"""

from typing import Union

import gqe.lib

from .execute import Context, MultiProcessContext
from .table_definition import CustomTableDefinitions, TableDefinitions, TPCHTableDefinitions


class Catalog:
    def __init__(self, context: Union[Context, MultiProcessContext]) -> None:
        self._context = context
        self._catalog = gqe.lib.Catalog(context._context)

    def register_tables(
        self,
        dataset: str,
        storage_kind: str = "pinned_memory",
        load_data_of_query: int = 0,
        identifier_type: gqe.lib.TypeId = gqe.lib.TypeId.int32,
        use_opt_char_type: bool = True,
        ddl_file_path: str = None,
        **kwargs,
    ) -> TableDefinitions:
        """
        Register dataset in the catalog.

        :arg dataset: Location of the dataset.
        :arg storage_kind: Storage kind for tables. Can be either `"pinned_memory"`,
            `"system_memory"`, `"numa_memory"`, `"device_memory"`, `"managed_memory"`,
            `"numa_pinned_memory"`, `"boost_shared_memory"`, `"numa_pool_memory"`,
            or `"parquet_file"`.
        :arg load_data_of_query: For in-memory storage,
            if `load_data_of_query = 0` loads entire dataset,
            else if `0 < load_data_of_query <= 22` loads table and columns required for the
            specific query
        :arg identifier_type: Can be either `gqe.lib.TypeId.int32` or `gqe.lib.TypeId.int64`
        :arg use_opt_char_type: If true, use optimized char type for single character columns.
        :arg ddl_file_path: Path to the DDL file for custom tables. It expects load_data_of_query to be 0.
        :arg kwargs: Additional keyword arguments are accepted and ignored. This allows callers
            to pass a CatalogContext via `**asdict(cat_ctx)` which may contain extra fields
            used elsewhere (e.g., compression settings for OptimizationParameters).
        """
        if ddl_file_path:
            print(f"Registering custom tables from DDL file: {ddl_file_path}")
            if not load_data_of_query == 0:
                raise ValueError(
                    "load_all_data must be True when registering custom tables from DDL file"
                )
            table_definitions = CustomTableDefinitions(ddl_file_path)
        else:
            table_definitions = TPCHTableDefinitions(identifier_type, use_opt_char_type)
        if storage_kind == "parquet_file":
            gqe.lib.register_tables_parquet(
                self._catalog,
                dataset,
                table_definitions.query_table_definitions(load_data_of_query),
            )
        elif storage_kind in [
            "pinned_memory",
            "system_memory",
            "numa_memory",
            "device_memory",
            "managed_memory",
            "numa_pinned_memory",
            "boost_shared_memory",
            "numa_pool_memory",
        ]:
            gqe.lib.register_tables_in_memory(
                self._context._context,
                self._catalog,
                dataset,
                table_definitions.query_table_definitions(load_data_of_query),
                storage_kind,
            )
        else:
            raise ValueError(f"Unrecognized storage kind: {storage_kind}")

        return table_definitions

    def load_substrait(
        self,
        substrait_file: str,
        optimized: bool = True,
        multiprocess_runtime_context: gqe.lib.MultiProcessRuntimeContext = None,
    ) -> gqe.lib.Relation:
        return gqe.lib.load_substrait(
            self._catalog,
            substrait_file,
            optimized,
            multiprocess_runtime_context,
        )
