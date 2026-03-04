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

from __future__ import annotations

from typing import TYPE_CHECKING

import gqe_bench.lib
from gqe_bench.relation import Relation

# Circular import of between execute and catalog
if TYPE_CHECKING:
    from gqe_bench.catalog import Catalog


class Context:
    def __init__(
        self,
        optimization_parameters: gqe_bench.lib.OptimizationParameters,
        debug_mem_usage: bool = False,
        cupti_metrics: list[str] | None = None,
    ):
        """
        Create a new context.

        :param optimization_parameters: Optimization parameters for query execution.
        :param debug_mem_usage: Enable debug memory usage tracking.
        :param cupti_metrics: The CUPTI range metrics to profile. If this argument is `None`, the profiler is completely disabled.
        """

        self._context = gqe_bench.lib.Context(
            optimization_parameters,
            debug_mem_usage,
            cupti_metrics,
        )

    def execute(
        self,
        catalog: Catalog,
        relation: Relation | gqe_bench.lib.Relation,
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

    def refresh_query_context(
        self, optimization_parameters: gqe_bench.lib.OptimizationParameters
    ) -> None:
        """
        Refresh the query context with new optimization parameters.

        :param optimization_parameters: New optimization parameters for query execution.
        """
        self._context.refresh_query_context(optimization_parameters)


class MultiProcessContext:
    def __init__(
        self,
        runtime_context: gqe_bench.lib.MultiProcessRuntimeContext,
        optimization_parameters: gqe_bench.lib.OptimizationParameters,
        scheduler_type: gqe_bench.lib.scheduler_type = gqe_bench.lib.scheduler_type.ROUND_ROBIN,
    ):
        """
        Create a new multi-process context.

        :param runtime_context: The multi-process runtime context.
        :param optimization_parameters: Optimization parameters for query execution.
        :param scheduler_type: The scheduler type for multi-process execution.
        """
        self._context = gqe_bench.lib.MultiProcessContext(
            runtime_context,
            optimization_parameters,
            scheduler_type,
        )

    def execute(
        self,
        catalog: Catalog,
        relation: Relation | gqe_bench.lib.Relation,
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

    def refresh_query_context(
        self, optimization_parameters: gqe_bench.lib.OptimizationParameters
    ) -> None:
        """
        Refresh the query context with new optimization parameters.

        :param optimization_parameters: New optimization parameters for query execution.
        """
        self._context.refresh_query_context(optimization_parameters)
