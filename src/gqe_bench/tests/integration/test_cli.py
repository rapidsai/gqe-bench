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

"""Integration tests for GqeCli — invokes the real gqe-cli binary."""

from pathlib import Path

import pytest

from gqe_bench.cli import GqeCli, GqeCliError
from gqe_bench.query_source import QuerySource
from gqe_bench.server import DEFAULT_SERVER_URL


class TestGqeCli:
    """End-to-end tests against the real gqe-cli binary."""

    def test_execute_sql_without_server(self, cli_bin: Path) -> None:
        """Executing SQL without a running server raises GqeCliError."""
        cli = GqeCli(cli_bin, DEFAULT_SERVER_URL)
        with pytest.raises(GqeCliError) as exc_info:
            cli.prepare(QuerySource.SQL, "SELECT 1;").execute()
        assert exc_info.value.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
