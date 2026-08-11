#!/usr/bin/env python3
#
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

"""Verifies every built artifact is accessible from the active install layout (wheel or editable)."""

import os

import pytest

from gqe_bench._artifacts import ALL
from gqe_bench.resources import Artifact


class TestArtifactResolution:
    """Verify every built artifact is present and (for binaries) executable."""

    @pytest.mark.parametrize("artifact", ALL, ids=lambda a: a.name)
    def test_locates(self, artifact: Artifact) -> None:
        path = artifact.locate()
        assert path is not None, f"{artifact.name} not built ({artifact.build_flag} OFF)"
        assert path.is_file(), f"{artifact.name} expected at {path} but missing"

    @pytest.mark.parametrize(
        "artifact",
        [a for a in ALL if a.executable],
        ids=lambda a: a.name,
    )
    def test_binary_executable(self, artifact: Artifact) -> None:
        path = artifact.locate()
        assert path is not None and path.is_file()
        assert os.access(path, os.X_OK), f"{artifact.name} at {path} not executable"
