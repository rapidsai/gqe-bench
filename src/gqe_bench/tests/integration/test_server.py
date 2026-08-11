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

"""Integration tests for GqeServer — starts the real gqe_node_manager subprocess."""

from pathlib import Path

import pytest

from gqe_bench.server import GqeServer


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Release the singleton slot before and after each test."""
    GqeServer._instance = None
    yield
    GqeServer._instance = None


class TestGqeServer:
    """Integration tests."""

    def test_start_stop(self, server_bin: Path, task_manager_bin: Path) -> None:
        """Server starts and stops cleanly via context manager."""
        with GqeServer(server_bin, task_manager_bin) as srv:
            assert srv._process is not None
            assert srv._process.poll() is None
        assert srv._process.poll() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
