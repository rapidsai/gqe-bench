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


class Catalog:
    def __init__(self) -> None:
        self._catalog = gqe.lib.Catalog()

    def register_tpch(
            self, dataset: str, storage: str = "parquet", num_row_groups: int = 8) -> None:
        """
        Register TPC-H dataset in the catalog.

        :arg dataset: Location of the TPC-H dataset.
        :arg storage: Can be either `"parquet"` for loading from Parquet files during execution, or
            `"memory"` for pre-copying dataset into CPU memory during registration time.
        :arg num_row_groups: Number of row groups for in-memory storage.
        """
        if storage == "parquet":
            gqe.lib.register_tpch_parquet(self._catalog, dataset)
        elif storage == "memory":
            gqe.lib.register_tpch_in_memory(self._catalog, dataset, num_row_groups)
        else:
            raise ValueError(f"Unrecognized storage: {storage}")
