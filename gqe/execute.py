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
        optimization_parameters: gqe.lib.OptimizationParameters,
        debug_mem_usage: bool = False,
        cupti_metrics: list[str] | None = None,
    ):
        """
        Create a new context.

        :param optimization_parameters: Optimization parameters for query execution.
        :param debug_mem_usage: Enable debug memory usage tracking.
        :param cupti_metrics: The CUPTI range metrics to profile. If this argument is `None`, the profiler is completely disabled.
        """

        self._context = gqe.lib.Context(
            optimization_parameters,
            debug_mem_usage,
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

    def refresh_query_context(
        self, optimization_parameters: gqe.lib.OptimizationParameters
    ) -> None:
        """
        Refresh the query context with new optimization parameters.

        :param optimization_parameters: New optimization parameters for query execution.
        """
        self._context.refresh_query_context(optimization_parameters)


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
        optimization_parameters: gqe.lib.OptimizationParameters,
        scheduler_type: gqe.lib.scheduler_type = gqe.lib.scheduler_type.ROUND_ROBIN,
    ):
        """
        Create a new multi-process context.

        :param runtime_context: The multi-process runtime context.
        :param optimization_parameters: Optimization parameters for query execution.
        :param scheduler_type: The scheduler type for multi-process execution.
        """
        self._context = gqe.lib.MultiProcessContext(
            runtime_context._context,
            optimization_parameters,
            scheduler_type,
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

    def refresh_query_context(
        self, optimization_parameters: gqe.lib.OptimizationParameters
    ) -> None:
        """
        Refresh the query context with new optimization parameters.

        :param optimization_parameters: New optimization parameters for query execution.
        """
        self._context.refresh_query_context(optimization_parameters)
