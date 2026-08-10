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

"""
Tests for the GqeCli wrapper.

Usage:
    pytest gqe_bench/tests/test_cli.py -v
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gqe_bench.cli import GqeCli, GqeCliError
from gqe_bench.query_source import QuerySource


def _mock_run_ok(*args: object, **kwargs: object) -> MagicMock:
    result = MagicMock()
    result.returncode = 0
    return result


class TestGqeCliCommands:
    """Tests for command construction (mocked subprocess)."""

    def test_sql_uses_stdin(self) -> None:
        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=_mock_run_ok) as mock:
            cli.prepare(QuerySource.SQL, "SELECT 1;").execute()
        cmd = mock.call_args[0][0]
        assert "--sql-file" in cmd
        assert "-" in cmd
        assert "--parquet" not in cmd
        assert mock.call_args[1]["input"] == b"SELECT 1;"

    def test_sql_with_parquet(self) -> None:
        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=_mock_run_ok) as mock:
            cli.prepare(QuerySource.SQL, b"SELECT 1;").with_parquet(Path("/out.parquet")).execute()
        cmd = mock.call_args[0][0]
        idx = cmd.index("--sql-file")
        assert cmd[idx + 1] == "-"
        assert mock.call_args[1]["input"] == b"SELECT 1;"
        assert "--parquet" in cmd
        assert "/out.parquet" in cmd

    def test_plan_with_parquet(self) -> None:
        """A handcoded plan streams its bytes over stdin; --physical-plan is '-'."""
        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        plan_bytes = b"\x00\x01\x02\x03fake-protobuf"
        with patch("gqe_bench.cli.subprocess.run", side_effect=_mock_run_ok) as mock:
            cli.prepare(QuerySource.HANDCODED, plan_bytes).with_parquet(
                Path("/out.parquet")
            ).execute()
        cmd = mock.call_args[0][0]
        idx = cmd.index("--physical-plan")
        assert cmd[idx + 1] == "-"
        assert mock.call_args[1]["input"] == plan_bytes
        assert "--parquet" in cmd
        assert "/out.parquet" in cmd

    def test_load_table_constructs_copy(self) -> None:
        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=_mock_run_ok) as mock:
            cli.load_table("lineitem", Path("/data/lineitem"))
        assert b"COPY lineitem FROM" in mock.call_args[1]["input"]

    def test_nonzero_returncode_raises(self) -> None:
        def fail_run(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.returncode = 1
            result.stderr = b"connection refused"
            return result

        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=fail_run):
            with pytest.raises(GqeCliError) as exc_info:
                cli.prepare(QuerySource.SQL, "SELECT 1;").execute()
            assert exc_info.value.returncode == 1
            assert "connection refused" in exc_info.value.stderr

    def test_timeout_raises_gqe_cli_error(self) -> None:
        def timeout_run(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd=["gqe-cli"], timeout=60)

        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=timeout_run):
            with pytest.raises(GqeCliError) as exc_info:
                cli.prepare(QuerySource.SQL, "SELECT 1;").with_timeout().execute()
            assert "timed out" in str(exc_info.value)


class TestGqeCliTimeoutPolicy:
    """prepare() defaults to no timeout; callers opt in via with_timeout()."""

    def test_load_schema_has_no_timeout(self) -> None:
        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=_mock_run_ok) as mock:
            cli.load_schema("CREATE TABLE t (id INT);")
        assert mock.call_args.kwargs.get("timeout") is None

    def test_load_table_has_no_timeout(self) -> None:
        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=_mock_run_ok) as mock:
            cli.load_table("nation", Path("/data/nation"))
        assert mock.call_args.kwargs.get("timeout") is None

    def test_drop_table_has_no_timeout(self) -> None:
        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=_mock_run_ok) as mock:
            cli.drop_table("nation")
        assert mock.call_args.kwargs.get("timeout") is None

    def test_prepare_default_has_no_timeout(self) -> None:
        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=_mock_run_ok) as mock:
            cli.prepare(QuerySource.SQL, "SELECT 1;").execute()
        assert mock.call_args.kwargs.get("timeout") is None

    def test_with_timeout_default_sets_hang_detection(self) -> None:
        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=_mock_run_ok) as mock:
            cli.prepare(QuerySource.SQL, "SELECT 1;").with_timeout().execute()
        assert mock.call_args.kwargs.get("timeout") == 60

    def test_with_timeout_sets_subprocess_timeout(self) -> None:
        cli = GqeCli(Path("/bin/cli"), "http://host:1234")
        with patch("gqe_bench.cli.subprocess.run", side_effect=_mock_run_ok) as mock:
            cli.prepare(QuerySource.SQL, "SELECT 1;").with_timeout(42).execute()
        assert mock.call_args.kwargs.get("timeout") == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
