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

"""Artifact-resource types for CMake-built gqe_bench payloads.

``gqe_bench._artifacts`` (CMake-generated) instantiates one ``Artifact`` per
built artifact; callers import the relevant instance and call ``.locate()``
for lax lookup (returns ``None`` when not present) or ``.require()`` for
strict lookup (raises ``ArtifactMissing``, which distinguishes "not built"
from "configured but file absent").
"""

from dataclasses import dataclass
from importlib.metadata import distribution
from pathlib import Path
from typing import Literal

_DIST_NAME = "gqe-bench"


def _install_path(rel: str) -> Path:
    """Resolve a path relative to the installed distribution root.

    Returns the install location regardless of install mode: site-packages
    for both editable (scikit-build-core's static install dir) and wheel
    installs. Differs from ``importlib.resources.files()``, which returns
    the source tree in editable mode.
    """
    return Path(distribution(_DIST_NAME).locate_file(rel))


@dataclass(frozen=True, slots=True)
class Artifact:
    """A CMake-built artifact in the installed ``gqe_bench`` package.

    Attributes:
        name:          Logical identifier (lowercase), e.g. ``"plugin"``.
        filename:      Distribution-relative path of the installed file
                       (e.g. ``"gqe_bench/_artifacts/bin/gqe-cli"``). Empty
                       string means the artifact's build flag was off at
                       CMake time — the "not built" sentinel.
        build_flag:    CMake option that gates production of this artifact.
        override_hint: CLI flag that overrides the resolved path, or empty
                       string if no override exists.
        executable:    Whether the artifact is run as a program, as opposed to
                       a shared library or a data file.
    """

    name: str
    filename: str
    build_flag: str
    override_hint: str = ""
    executable: bool = False

    def locate(self) -> Path | None:
        """Return the artifact's path if present in the install, else ``None``.

        Never raises for the "not built" or "not found" cases. Environmental
        failures (unloadable package, filesystem errors) propagate.
        """
        if not self.filename:
            return None
        p = _install_path(self.filename)
        return p if p.is_file() else None

    def require(self) -> Path:
        """Return the artifact's path or raise :class:`ArtifactMissing`.

        Distinguishes ``"not_built"`` (empty :attr:`filename`) from
        ``"missing"`` (file absent at expected location) via the exception's
        ``reason`` attribute and message.
        """
        if not self.filename:
            raise ArtifactMissing(self, reason="not_built")
        p = _install_path(self.filename)
        if not p.is_file():
            raise ArtifactMissing(self, reason="missing", expected=p)
        return p


class ArtifactMissing(RuntimeError):
    """Raised by :meth:`Artifact.require` when the artifact isn't available.

    ``reason`` distinguishes the two failure modes: ``"not_built"`` means the
    ``build_flag`` was off at CMake time (empty :attr:`Artifact.filename`),
    so the artifact was never produced; ``"missing"`` means CMake configured
    the artifact but the file isn't where it should be (build partially ran,
    file deleted, etc.).
    """

    def __init__(
        self,
        artifact: Artifact,
        reason: Literal["not_built", "missing"],
        expected: Path | None = None,
    ):
        self.artifact = artifact
        self.reason = reason
        self.expected = expected
        if reason == "not_built":
            msg = f"{artifact.name} not built (set {artifact.build_flag}=ON"
            if artifact.override_hint:
                msg += f" or override via {artifact.override_hint}"
            msg += ")."
        else:
            msg = f"{artifact.name} expected at {expected} but not found — rebuild"
            if artifact.override_hint:
                msg += f" or override via {artifact.override_hint}"
            msg += "."
        super().__init__(msg)
