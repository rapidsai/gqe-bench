# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled physical-plan protobuf modules.

The ``.proto`` files come from the gqe checkout the engine is built from, so the
wire format cannot disagree with the client reading it. CMake compiles them and
installs the ``*_pb2`` modules alongside this one; the names below alias them.

Resolution is deferred to first attribute access, so a ``GQE_FETCH_ENGINE=OFF``
build still imports this module and raises ``ModuleNotFoundError`` only where one
of the names below is read.
"""

import importlib
from types import ModuleType

_MODULES = {
    "data_type": "data_type_pb2",
    "expression": "expression_pb2",
    "physical_plan": "physical_plan_pb2",
    "result": "result_pb2",
}


def __getattr__(name: str) -> ModuleType:
    """Resolve one of the four proto module aliases, caching it in globals().

    Raises:
        AttributeError: ``name`` is not one of the aliases.
        ModuleNotFoundError: The modules were not generated, i.e. the package
            was built with ``GQE_FETCH_ENGINE=OFF``.
    """
    target = _MODULES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__package__}.{target}")
    globals()[name] = module
    return module
