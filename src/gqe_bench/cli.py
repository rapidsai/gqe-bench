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

"""GqeCli wrapper — thin interface around the gqe-cli binary.

Executions default to no timeout. Query execution sites that want
hang-detection opt in via ``GqeCliExecution.with_timeout(seconds)``;
data-load sites (load_schema, load_table, drop_table) leave the default in
place because real-dataset loads can run for many minutes.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from gqe_bench.query_source import QuerySource
from gqe_bench.server import DEFAULT_SERVER_URL

logger = logging.getLogger(__name__)

_QUERY_TIMEOUT = 60


def _format_stdin_for_log(stdin_data: str | bytes | None) -> str:
    """Render the stdin content for the debug log: SQL text inline, plan bytes as a size."""
    if stdin_data is None:
        return ""
    if isinstance(stdin_data, bytes):
        return f"\n    stdin: <plan: {len(stdin_data)} bytes>"
    return f"\n    stdin: {stdin_data}"


class GqeCliError(RuntimeError):
    """Raised when a gqe-cli invocation fails."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"gqe-cli exited with code {returncode}\n"
            f"Command: {' '.join(str(c) for c in cmd)}\n"
            f"Stderr: {stderr}"
        )


class GqeCliExecution:
    """Builder for a gqe-cli invocation.

    Fluent methods (with_*) accumulate options and return self; execute()
    fires the subprocess with whatever was accumulated.
    """

    def __init__(
        self,
        cli: GqeCli,
        source: QuerySource,
        content: str | bytes,
    ) -> None:
        self._cli = cli
        self._timeout: int | None = None
        match source:
            case QuerySource.SQL:
                self._args = ["--sql-file", "-"]
            case QuerySource.HANDCODED:
                self._args = ["--physical-plan", "-"]
            case _:
                raise ValueError(f"unsupported query source: {source}")
        self._stdin_data: str | bytes = content

    def with_parquet(self, path: Path) -> GqeCliExecution:
        """Direct query results to a parquet file."""
        self._args += ["--parquet", str(path)]
        return self

    def with_timeout(self, seconds: int = _QUERY_TIMEOUT) -> GqeCliExecution:
        """Set a per-execution timeout in seconds (default: hang-detection)."""
        self._timeout = seconds
        return self

    def execute(self) -> None:
        """Fire the subprocess with accumulated options."""
        self._cli._run(self._args, stdin_data=self._stdin_data, timeout=self._timeout)


class GqeCli:
    """Wrapper around gqe-cli subprocess calls."""

    def __init__(self, cli_bin: Path | None, server_url: str = DEFAULT_SERVER_URL) -> None:
        """Bind a gqe-cli binary to a server URL.

        Raises ValueError if ``cli_bin`` is None.
        """
        if cli_bin is None:
            raise ValueError("CLI binary path not provided")
        self._bin = cli_bin
        self._server_url = server_url

    def _run(
        self,
        extra_args: list[str],
        stdin_data: str | bytes | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        """Run gqe-cli with the given args.

        ``stdin_data`` is written to the subprocess's stdin as bytes — SQL text
        is UTF-8 encoded, plan bytes are passed through unchanged. Raises
        GqeCliError on subprocess timeout or non-zero exit.
        """
        cmd: list[str] = [str(self._bin), "--server-url", self._server_url, *extra_args]
        input_bytes = stdin_data.encode("utf-8") if isinstance(stdin_data, str) else stdin_data
        logger.debug("gqe-cli cmd: %s%s", " ".join(cmd), _format_stdin_for_log(stdin_data))
        try:
            result = subprocess.run(
                cmd,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise GqeCliError(cmd, -1, f"timed out after {timeout}s")
        if result.returncode != 0:
            raise GqeCliError(
                cmd, result.returncode, (result.stderr or b"").decode("utf-8", errors="replace")
            )
        return result

    def prepare(self, source: QuerySource, content: str | bytes) -> GqeCliExecution:
        """Build a query execution. Chain fluent with_* options, then .execute()."""
        return GqeCliExecution(self, source, content)

    def load_schema(self, sql: str) -> None:
        """Load a SQL schema (CREATE TABLE statements) from a string."""
        self.prepare(QuerySource.SQL, sql).execute()

    def load_table(
        self,
        table_name: str,
        data_dir: Path,
        columns: list[str] | None = None,
    ) -> None:
        """Load a table from parquet files via COPY.

        When ``columns`` is provided, emits a column-list COPY so the
        engine reads only those columns from parquet; otherwise loads
        every column declared in the DDL.
        """
        col_list = f"({','.join(columns)})" if columns else ""
        self.prepare(
            QuerySource.SQL,
            f"COPY {table_name}{col_list} FROM '{data_dir}' (FORMAT parquet)",
        ).execute()

    def drop_table(self, table_name: str) -> None:
        """Drop a table from the server (no-op if it doesn't exist)."""
        self.prepare(QuerySource.SQL, f"DROP TABLE IF EXISTS {table_name}").execute()

    def set_optimization_parameter(self, name: str, value: object) -> None:
        """Set an engine optimization parameter on this session via SQL ``SET``."""
        self.prepare(QuerySource.SQL, f"SET {name} TO {value}").execute()
