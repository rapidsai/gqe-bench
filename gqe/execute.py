# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional, Union  # Not needed with Python>=3.10


class Context:
    def __init__(self,
                 max_num_workers: int = 1,
                 max_num_partitions: int = 8,
                 read_zero_copy_enable: bool = False):
        self._context = gqe.lib.Context(max_num_workers, max_num_partitions, read_zero_copy_enable)

    def execute(
            self,
            catalog: Catalog,
            relation: Union[Relation, gqe.lib.Relation],
            output_path: Optional[str]) -> float:
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
