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

from abc import ABC, abstractmethod

from gqe_bench.relation import Relation
from gqe_bench.table_definition import TPCHTableDefinitions


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
