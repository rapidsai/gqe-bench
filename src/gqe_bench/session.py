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
import tempfile
from argparse import Namespace
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from multiprocessing import shared_memory
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from gqe_bench._agg_protocol_sizes import K_HEADER_EXPERIMENT_ID_OFFSET, K_HEADER_RUN_NUMBER_OFFSET
from gqe_bench._artifacts import (
    CLI,
    GQE_BENCH_SHM_NAME_ENV,
    GQE_BENCH_SHM_SIZE_ENV,
    NODE_MANAGER,
    TASK_MANAGER,
)
from gqe_bench.cli import GqeCli, GqeCliError
from gqe_bench.profiling import NoopProfiler, NsysError, NsysProfiler, Profiler
from gqe_bench.query_source import QuerySource
from gqe_bench.schema import (
    DATA_INFO_MAPPING,
    LEGACY_TASKGRAPH_EXECUTION_BACKEND,
    QUERY_PARAMS_MAPPING,
)
from gqe_bench.server import (
    GqeServer,
    ServerAlreadyRunning,
    ServerStartFailed,
)
from gqe_bench.suites.base import append_storage_kind_clause, build_external_table_ddl, narrow_ddl

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
_LOAD_NON_SET_FIELDS: frozenset[str] = frozenset({"storage_device_kind", "shmem_policy"})

# Engine-side statistics system tables, registered by the node manager
# (gqe/include/gqe/node_manager/stats_tables.hpp).
_TABLE_STATS: Final = "_gqe_table_stats"
_COLUMN_STATS: Final = "_gqe_column_stats"


class RestartRequired(RuntimeError):
    """Caller must tear down the session and start a fresh one."""


class QueryFailed(RuntimeError):
    """Single query failed; record the failure and proceed to the next query."""


class GqeSession:
    """Context manager owning one gqe_node_manager subprocess + one CLI.

    Usage::

        with GqeSession(args, env, schema_ddl, tables, data_info=data_info) as session:
            session.load_data()
            session.execute_query(query, output_path, experiment_id, run_number)

    Raises only RestartRequired / QueryFailed. Other exceptions propagate
    unchanged.
    """

    def __init__(
        self,
        args: Namespace,
        env: dict[str, str],
        schema_ddl: str,
        tables: list[tuple[str, Path]],
        data_info: DataInfo,
    ) -> None:
        self._args = args
        self._env = env
        self._schema_ddl = schema_ddl
        self._tables = tables
        # NoopProfiler without --nsys, so neither this class nor GqeServer has
        # to check whether profiling is on.
        self._profiler: Profiler = (
            NsysProfiler(
                args.nsys,
                output_dir=args.nsys_output_dir,
                prefix=args.nsys_prefix,
                trace=args.nsys_trace,
                launch_args=args.nsys_launch_args,
                extra_args=args.nsys_args,
            )
            if args.nsys
            else NoopProfiler()
        )
        self._data_info = data_info
        self._stack: ExitStack | None = None
        self._server: GqeServer | None = None
        self._cli: GqeCli | None = None
        self._agg_shm: shared_memory.SharedMemory | None = None

    @contextmanager
    def profile(self, label: str) -> Iterator[None]:
        """Collect one profile named after ``label`` around the block.

        Everything executed inside the block lands in one report. A failure to
        arm propagates without closing anything, since there is nothing open.
        """
        self.start_profile(label)
        try:
            yield
        finally:
            self.stop_profile()

    def start_profile(self, label: str) -> None:
        """Begin a profile named after ``label``; no-op without --nsys.

        The caller owns the window because only it knows what belongs in one
        report; every execution between this and ``stop_profile`` lands in it.
        On NsysError, raises RestartRequired so the caller tears down and
        retries with a fresh session.
        """
        try:
            self._profiler.start(label)
        except NsysError as e:
            raise RestartRequired(str(e)) from e

    def stop_profile(self) -> None:
        """Close the profile opened by ``start_profile``; no-op without --nsys."""
        self._profiler.stop()

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
                        profiler=self._profiler,
                        shmem_policy=self._data_info.shmem_policy,
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
    ) -> None:
        """Drop existing tables, reload schema, reload table data.

        ``required_columns`` selects what to load. ``LOAD_ALL_COLUMNS`` loads
        every column of every table; a per-table map narrows the schema and
        skips absent tables. In-memory storage COPYs the narrowed columns and
        keeps skipped tables empty; ``parquet_file`` storage reads tables
        directly from location instead of COPY, dropping skipped tables from
        the schema entirely.

        Emits per-table INFO so a human watching ``tail -f`` sees the load
        advance — a SF1K load runs minutes and silence looks like a hang.
        Any CLI failure during load is unrecoverable for this DataInfo —
        raises RestartRequired so the caller tears down the session.
        """
        try:
            # Query parameters persist for the Flight SQL session. A previous
            # query may have selected QEP, but table registration and COPY must
            # stay on the supported legacy path.
            # TODO: Remove this reset once data loading no longer requires legacy_task_graph.
            self._cli.set_optimization_parameter(
                "execution_backend", LEGACY_TASKGRAPH_EXECUTION_BACKEND
            )

            # Serialization is engine-driven via optimization parameters: CREATE TABLE
            # reconstructs each in-memory table from its snapshot when
            # `deserialize_tables_from_disk` is set, and each COPY snapshots the table it populated
            # when `serialize_tables_to_disk` is set. These must be set before load_schema.
            #
            # `serialized_dataset_location` is sent unconditionally.
            self._cli.set_optimization_parameter(
                "serialized_dataset_location", str(self._args.dataset)
            )
            if self._args.serialize_to_disk:
                self._cli.set_optimization_parameter("deserialize_tables_from_disk", False)
                self._cli.set_optimization_parameter("serialize_tables_to_disk", True)
            elif self._args.deserialize_from_disk:
                self._cli.set_optimization_parameter("serialize_tables_to_disk", False)
                self._cli.set_optimization_parameter("deserialize_tables_from_disk", True)

            for f in DATA_INFO_MAPPING.sweep_fields:
                if f in _LOAD_NON_SET_FIELDS:
                    continue
                self._cli.set_optimization_parameter(
                    _LOAD_ENGINE_NAME_OVERRIDES.get(f, f),
                    getattr(self._data_info, f),
                )

            for name, _ in self._tables:
                self._cli.drop_table(name)

            # gqe-cli derives each table's COPY columns from its registered
            # schema, so narrowing the CREATE TABLE is what projects the load.
            schema_ddl = self._schema_ddl
            if required_columns is not LOAD_ALL_COLUMNS:
                schema_ddl = narrow_ddl(self._schema_ddl, required_columns)
            referenced = {
                name
                for name, _ in self._tables
                if required_columns is LOAD_ALL_COLUMNS or name in required_columns
            }

            is_parquet = self._data_info.storage_device_kind == "parquet_file"
            if is_parquet:
                locations = {
                    name: table_dir for name, table_dir in self._tables if name in referenced
                }
                schema_ddl = build_external_table_ddl(schema_ddl, locations)
            else:
                schema_ddl = append_storage_kind_clause(
                    schema_ddl, self._data_info.storage_device_kind
                )
            self._cli.load_schema(schema_ddl)

            for name, table_dir in self._tables:
                if name not in referenced:
                    logger.debug("Skipping table: %s (not referenced)", name)
                    continue
                if is_parquet:
                    logger.debug("Loading external table: %s (reads directly from location)", name)
                elif self._args.deserialize_from_disk:
                    # Table is reconstructed from its on-disk snapshot, skip copy.
                    logger.debug("Deserialized table from snapshot: %s", name)
                else:
                    logger.debug("Loading table: %s", name)
                    self._cli.load_table(name, table_dir)
        except GqeCliError as e:
            raise RestartRequired(str(e)) from e

    def read_storage_stats(self) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Read the engine's storage statistics for the data currently loaded.

        ``_gqe_table_stats`` and ``_gqe_column_stats`` are queryable system
        tables, so this is an ordinary SELECT rather than a side channel.
        Returns ``(table_stats, column_stats)``.

        On GqeCliError:
          * server alive -> None (caller records no statistics).
          * server dead  -> RestartRequired (caller restarts).

        An engine that does not expose the tables costs the statistics and
        nothing else, so an older build degrades to "no statistics recorded"
        rather than failing a sweep that is otherwise fine. A dead server is
        not that case: every query in the group would fail against it.

        Call after ``load_data``: compression is decided at write time, so the
        numbers describe the load and do not change between parameter
        combinations over the same data.
        """
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = {}
                for key, table in (("table", _TABLE_STATS), ("column", _COLUMN_STATS)):
                    path = Path(tmp) / f"{key}.parquet"
                    self._cli.prepare(
                        QuerySource.SQL, f"SELECT * FROM {table}".encode()
                    ).with_timeout().with_parquet(path).execute()
                    out[key] = pd.read_parquet(path)
            return out["table"], out["column"]
        except GqeCliError as e:
            if not self._server.is_alive():
                raise RestartRequired(str(e)) from e
            # Not fatal: the run itself is unaffected by missing statistics.
            logger.warning("Storage statistics unavailable (%s); continuing without them", e)
            return None

    def execute_query(
        self,
        query: Query,
        output_path: Path,
        experiment_id: int,
        run_number: int = 0,
        qp: QueryParams | None = None,
    ) -> None:
        """Run the query via gqe-cli, writing parquet results to output_path.

        ``experiment_id`` and ``run_number`` are posted into the shm
        segment for the duration of the call; the per-query rows committed
        in the on-pop callback are keyed on them.

        On GqeCliError, from setting the query parameters or from the query itself:
          * server alive -> QueryFailed (caller records + next query).
          * server dead  -> RestartRequired (caller records + restart).

        Output validation is the caller's concern.
        """
        if qp is not None:
            try:
                for f in QUERY_PARAMS_MAPPING.sweep_fields:
                    self._cli.set_optimization_parameter(f, getattr(qp, f))
            except GqeCliError as e:
                raise self._classify_cli_error(e) from e
        self._begin_query(experiment_id, run_number)
        try:
            self._cli.prepare(query.source, query.content).with_timeout().with_parquet(
                output_path
            ).execute()
        except GqeCliError as e:
            raise self._classify_cli_error(e) from e
        finally:
            self._end_query()

    def _classify_cli_error(self, e: GqeCliError) -> QueryFailed | RestartRequired:
        """Map a gqe-cli failure to QueryFailed if the server is alive, else RestartRequired."""
        if self._server.is_alive():
            return QueryFailed(str(e))
        return RestartRequired(str(e))

    def _begin_query(self, experiment_id: int, run_number: int) -> None:
        """Re-zero the shm segment and post the request into the header.

        The full-segment zero resets the per-slot generation counters
        between queries. The run number is stored before the experiment id:
        a non-zero id at ``K_HEADER_EXPERIMENT_ID_OFFSET`` is what the next
        ``execute_plan`` push acts on, so it is written last.
        """
        if self._agg_shm is None:
            return
        np.frombuffer(self._agg_shm.buf, dtype=np.uint8).fill(0)
        np.frombuffer(
            self._agg_shm.buf, dtype=np.int32, count=1, offset=K_HEADER_RUN_NUMBER_OFFSET
        )[0] = run_number
        np.frombuffer(
            self._agg_shm.buf, dtype=np.int32, count=1, offset=K_HEADER_EXPERIMENT_ID_OFFSET
        )[0] = experiment_id

    def _end_query(self) -> None:
        """Clear the request slot back to the sentinel.

        Any subsequent ``execute_plan`` push that fires before the next
        ``_begin_query`` reads the sentinel and is skipped.
        """
        if self._agg_shm is None:
            return
        np.frombuffer(
            self._agg_shm.buf, dtype=np.int32, count=1, offset=K_HEADER_EXPERIMENT_ID_OFFSET
        )[0] = 0
