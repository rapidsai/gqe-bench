# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC, abstractmethod
from gqe.relation import Relation
from gqe.table_definition import TPCHTableDefinitions


class Query(ABC):
    """Abstract physical query plan generator."""

    def __init__(self, **kwargs):
        """
        Creates a a new physical query plan generator.

        Parameters:
          kwargs: Subclasses may take keyword arguments.
        """

        pass

    @abstractmethod
    def root_relation(self, table_defs: TPCHTableDefinitions) -> Relation:
        """
        Creates a phyiscal query plan.

        :param table_defs the table defintions to get the column types for better printing physical plan
        Returns:
          Relation: The GQE root relation of the physical query plan.
        """

        pass
