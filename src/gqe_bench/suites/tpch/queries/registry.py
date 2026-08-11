# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H handcoded query registry."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gqe_bench.physical_plan.relation import Relation
    from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def lookup(query_name: str) -> Callable[[TpchTableSchema, float], Relation]:
    """Return the build_plan function for a handcoded TPC-H query."""
    module_name = f"gqe_bench.suites.tpch.queries.q{query_name}"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"No handcoded query module for '{query_name}'") from None
    return mod.build_plan
