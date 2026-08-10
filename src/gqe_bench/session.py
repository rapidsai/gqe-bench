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

"""GqeSession — owns the server + CLI lifecycle.

Translates server-domain errors (GqeServer start failures and singleton
rejects) and CLI-domain errors (GqeCliError with a liveness check) into
two typed recovery signals: RestartRequired and QueryFailed. Output
validation is the caller's concern and lives in gqe_bench.validate; this
module does not touch parquet files.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from contextlib import ExitStack
from multiprocessing import shared_memory
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Final

import numpy as np

from gqe_bench._agg_protocol_sizes import K_HEADER_REQUEST_OFFSET
from gqe_bench._artifacts import (
    CLI,
    GQE_BENCH_SHM_NAME_ENV,
    GQE_BENCH_SHM_SIZE_ENV,
    NODE_MANAGER,
    TASK_MANAGER,
)
from gqe_bench.cli import GqeCli, GqeCliError
from gqe_bench.schema import DATA_INFO_MAPPING, QUERY_PARAMS_MAPPING
from gqe_bench.server import (
    GqeServer,
    ServerAlreadyRunning,
    ServerStartFailed,
)
from gqe_bench.suites.base import narrow_ddl

if TYPE_CHECKING:
    from gqe_bench.schema import DataInfo, Query, QueryParams

logger = logging.getLogger(__name__)


# Sentinel for load_data's ``required_columns``: load every column of every
# table (no projection). A named alias so call sites and the projection check
# read as "load all columns" instead of a bare ``None``.
LOAD_ALL_COLUMNS: Final = None


_LOAD_ENGINE_NAME_OVERRIDES: dict[str, str] = {
    "num_row_groups": "num_partitions",
}
_LOAD_BENCH_ONLY: frozenset[str] = frozenset({"storage_device_kind"})


class RestartRequired(RuntimeError):
    """Caller must tear down the session and start a fresh one."""


class QueryFailed(RuntimeError):
    """Single query failed; record the failure and proceed to the next query."""


class GqeSession:
    """Context manager owning one gqe_node_manager subprocess + one CLI.

    Usage::

        with GqeSession(args, env, schema_ddl, tables) as session:
            session.load_data()
            session.execute_query(query, output_path)

    Raises only RestartRequired / QueryFailed. Other exceptions propagate
    unchanged.
    """

    def __init__(
        self,
        args: Namespace,
        env: dict[str, str],
        schema_ddl: str,
        tables: list[tuple[str, Path]],
    ) -> None:
        self._args = args
        self._env = env
        self._schema_ddl = schema_ddl
        self._tables = tables
        self._stack: ExitStack | None = None
        self._server: GqeServer | None = None
        self._cli: GqeCli | None = None
        self._agg_shm: shared_memory.SharedMemory | None = None

    def __enter__(self) -> GqeSession:
        """Start the server, then build a CLI client bound to it.

        ServerAlreadyRunning and ServerStartFailed are translated into
        RestartRequired so the caller can match on the typed recovery signal;
        other exceptions propagate unchanged.
        """
        self._stack = ExitStack()
        try:
            # User-supplied --*-bin flags take precedence; otherwise require
            # the package-resolved artifact. ArtifactMissing surfaces at this
            # spawn site (not at parse time) so tools like --help don't crash
            # on a partial build.
            server_bin = self._args.server_bin or NODE_MANAGER.require()
            task_manager_bin = self._args.task_manager_bin or TASK_MANAGER.require()
            cli_bin = self._args.cli_bin or CLI.require()
            try:
                self._create_agg_shm()
                self._server = self._stack.enter_context(
                    GqeServer(
                        server_bin,
                        task_manager_bin,
                        num_gpus=self._args.num_gpus,
                        env=self._env,
                    )
                )
            except (ServerAlreadyRunning, ServerStartFailed) as e:
                raise RestartRequired(str(e)) from e
            except OSError as e:
                raise RestartRequired(f"agg_protocol setup failed: {e}") from e
            self._cli = GqeCli(cli_bin, self._args.server_url)
            return self
        except BaseException:
            # Anything unhappy on the way out — typed RestartRequired, an
            # ArtifactMissing from the .require() calls, a KeyboardInterrupt,
            # any unexpected exception — must close the ExitStack so resources
            # registered before the throw (currently the agg_protocol unlink
            # callback) fire their cleanup. Without this the segment leaks
            # because the user's `with` block is never entered and __exit__
            # is never called.
            self._stack.close()
            raise

    def _create_agg_shm(self) -> None:
        """Create the multi-GPU aggregation shm segment ahead of node_manager.

        Sized by ``recording._build_plugin_env``; the plugin's
        ``agg_protocol::attach`` re-asserts the size on attach for drift
        detection. close + unlink callbacks are registered on the
        ExitStack so cleanup runs in LIFO order (server stop -> shm
        close -> shm unlink) on exit.
        """
        shm_name = self._env.get(GQE_BENCH_SHM_NAME_ENV, "")
        shm_size_str = self._env.get(GQE_BENCH_SHM_SIZE_ENV, "")
        if not shm_name or not shm_size_str:
            return
        self._agg_shm = shared_memory.SharedMemory(
            create=True, size=int(shm_size_str), name=shm_name
        )
        self._stack.callback(self._agg_shm.unlink)
        self._stack.callback(self._agg_shm.close)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Tear down the server via the ExitStack.

        Delegates to ExitStack.__exit__, which invokes GqeServer.__exit__
        unconditionally — that's where the subprocess stop and singleton
        release happen. GqeServer.__exit__ returns None, so the stack
        suppresses no exceptions.
        """
        if self._stack is not None:
            self._stack.__exit__(exc_type, exc_val, exc_tb)

    def load_data(
        self,
        required_columns: dict[str, set[str]] | None = LOAD_ALL_COLUMNS,
        data_info: DataInfo | None = None,
    ) -> None:
        """Drop existing tables, reload schema, reload table data.

        ``required_columns`` selects what to load. ``LOAD_ALL_COLUMNS`` loads
        every table with every column. A per-table column map projects the
        load: the reloaded schema is narrowed to those columns (so gqe-cli's
        COPY, which it derives from the registered schema, reads only them),
        and tables absent from the map are skipped (``load_schema`` still
        creates them, as empty tables).

        Emits per-table INFO so a human watching ``tail -f`` sees the load
        advance — a SF1K load runs minutes and silence looks like a hang.
        Any CLI failure during load is unrecoverable for this DataInfo —
        raises RestartRequired so the caller tears down the session.
        """
        try:
            if data_info is not None:
                for f in DATA_INFO_MAPPING.sweep_fields:
                    if f in _LOAD_BENCH_ONLY:
                        continue
                    self._cli.set_optimization_parameter(
                        _LOAD_ENGINE_NAME_OVERRIDES.get(f, f),
                        getattr(data_info, f),
                    )

            for name, _ in self._tables:
                self._cli.drop_table(name)

            # gqe-cli derives each table's COPY columns from its registered
            # schema, so narrowing the CREATE TABLE is what projects the load.
            schema_ddl = self._schema_ddl
            if required_columns is not LOAD_ALL_COLUMNS:
                schema_ddl = narrow_ddl(self._schema_ddl, required_columns)
            self._cli.load_schema(schema_ddl)

            for name, table_dir in self._tables:
                if required_columns is not LOAD_ALL_COLUMNS and name not in required_columns:
                    logger.debug("Skipping table: %s (not referenced)", name)
                    continue
                logger.debug("Loading table: %s", name)
                self._cli.load_table(name, table_dir)
        except GqeCliError as e:
            raise RestartRequired(str(e)) from e

    def execute_query(
        self,
        query: Query,
        output_path: Path,
        experiment_id: int,
        qp: QueryParams | None = None,
    ) -> None:
        """Run the query via gqe-cli, writing parquet results to output_path.

        ``experiment_id`` is posted into the shm segment for the
        duration of the call; the per-query rows committed in the
        on-pop callback are keyed on it.

        On GqeCliError:
          * server alive -> QueryFailed (caller records + next query).
          * server dead  -> RestartRequired (caller records + restart).

        Output validation is the caller's concern.
        """
        if qp is not None:
            for f in QUERY_PARAMS_MAPPING.sweep_fields:
                self._cli.set_optimization_parameter(f, getattr(qp, f))
        self._begin_query(experiment_id)
        try:
            self._cli.prepare(query.source, query.content).with_timeout().with_parquet(
                output_path
            ).execute()
        except GqeCliError as e:
            if self._server.is_alive():
                raise QueryFailed(str(e)) from e
            raise RestartRequired(str(e)) from e
        finally:
            self._end_query()

    def _begin_query(self, experiment_id: int) -> None:
        """Re-zero the shm segment and post ``experiment_id`` into the header.

        The full-segment zero resets the per-slot generation counters
        between queries; the int32 store at ``K_HEADER_REQUEST_OFFSET``
        is the request observed at the next ``execute_plan`` push.
        """
        if self._agg_shm is None:
            return
        np.frombuffer(self._agg_shm.buf, dtype=np.uint8).fill(0)
        np.frombuffer(self._agg_shm.buf, dtype=np.int32, count=1, offset=K_HEADER_REQUEST_OFFSET)[
            0
        ] = experiment_id

    def _end_query(self) -> None:
        """Clear the request slot back to the sentinel.

        Any subsequent ``execute_plan`` push that fires before the next
        ``_begin_query`` reads the sentinel and is skipped.
        """
        if self._agg_shm is None:
            return
        np.frombuffer(self._agg_shm.buf, dtype=np.int32, count=1, offset=K_HEADER_REQUEST_OFFSET)[
            0
        ] = 0
