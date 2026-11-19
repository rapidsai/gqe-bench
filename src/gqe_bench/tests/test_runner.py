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
Tests for the benchmark runner and GqeSession.

Usage:
    pytest gqe_bench/tests/test_runner.py -v
"""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from gqe_bench.cli import GqeCliError
from gqe_bench.profiling import NoopProfiler, NsysError
from gqe_bench.query_source import QuerySource
from gqe_bench.recording import DataDimensions, SweepContext, SystemIds
from gqe_bench.schema import DataLoadGroup
from gqe_bench.session import GqeSession, QueryFailed, RestartRequired
from gqe_bench.suites.tpch import TpchSuite
from gqe_bench.tests.helpers import make_data_info, make_query, make_query_params
from gqe_bench.validate import ValidationFailed


def _make_sweep(db_mgr: MagicMock | None = None) -> SweepContext:
    """Build a SweepContext suitable for _run_group tests."""
    return SweepContext(
        db_mgr=db_mgr if db_mgr is not None else MagicMock(),
        system_ids=SystemIds(1, 1, 1),
        gpu_info_ids=[],
        env={},
        suite_name=TpchSuite.NAME,
        repeat=1,
        records_timing=True,
    )


def _mock_gqe_session() -> MagicMock:
    """Mock GqeSession for _run_group tests, with storage statistics off.

    ``MagicMock(spec=...)`` would otherwise make ``read_storage_stats`` return a
    MagicMock, which _run_group treats as a real ``(table_stats, column_stats)``
    pair. Tests that exercise the recording path set the return value instead.
    """
    session = MagicMock(spec=GqeSession)
    session.read_storage_stats.return_value = None
    return session


def _session_with_mocks(
    server_alive: bool = True,
    *,
    serialize_to_disk: bool = False,
    deserialize_from_disk: bool = False,
) -> tuple[GqeSession, MagicMock, MagicMock]:
    """Build a GqeSession whose server + cli are mocks, bypassing __enter__.

    The serialize flags are pinned rather than left to the MagicMock because every attribute of a
    bare mock is truthy. Left unset, load_data would take the deserialize path and skip every
    load_table call, which is the behaviour a real run only has when the flag is actually on.
    """
    session = GqeSession.__new__(GqeSession)
    session._args = MagicMock()
    session._args.serialize_to_disk = serialize_to_disk
    session._args.deserialize_from_disk = deserialize_from_disk
    session._env = {}
    session._data_info = make_data_info(storage_device_kind="shared_segment_memory")
    session._schema_ddl = "CREATE TABLE t (id INT);"
    session._tables = []
    session._profiler = NoopProfiler()
    session._stack = None
    session._server = MagicMock()
    session._server.is_alive.return_value = server_alive
    session._cli = MagicMock()
    session._agg_shm = None
    return session, session._cli, session._server


def test_session_uses_data_info_policy_at_startup() -> None:
    args = Namespace(
        server_bin=Path("/bin/server"),
        task_manager_bin=Path("/bin/task_manager"),
        cli_bin=Path("/bin/cli"),
        num_gpus=1,
        server_url="http://127.0.0.1:50051",
        nsys=None,
    )
    with (
        patch("gqe_bench.session.GqeServer") as server,
        patch("gqe_bench.session.GqeCli"),
        GqeSession(args, {}, "", [], data_info=make_data_info(shmem_policy="future_engine_policy")),
    ):
        assert server.call_args.kwargs["shmem_policy"] == "future_engine_policy"


class TestGqeSessionExecuteQuery:
    """Tests for GqeSession.execute_query (source-agnostic)."""

    def test_calls_cli_prepare_with_parquet(self) -> None:
        session, cli, _ = _session_with_mocks()
        query = make_query(name="1", source=QuerySource.SQL, content=b"SELECT 1")
        output_path = Path("/tmp/out.parquet")
        session.execute_query(query, output_path, experiment_id=1)
        cli.prepare.assert_called_once_with(query.source, query.content)
        with_timeout = cli.prepare.return_value.with_timeout
        with_timeout.assert_called_once()
        with_timeout.return_value.with_parquet.assert_called_once_with(output_path)
        with_timeout.return_value.with_parquet.return_value.execute.assert_called_once()

    def test_plan_query_works_same_as_sql(self) -> None:
        session, cli, _ = _session_with_mocks()
        query = make_query(name="1", source=QuerySource.HANDCODED, content=b"AAAA")
        output_path = Path("/tmp/out.parquet")
        session.execute_query(query, output_path, experiment_id=1)
        cli.prepare.assert_called_once_with(query.source, query.content)

    def test_sets_execution_backend_before_query(self) -> None:
        """Set the backend per query so one persistent session can sweep backend values."""
        session, cli, _ = _session_with_mocks()
        query = make_query(name="6", source=QuerySource.SQL, content=b"SELECT 1")
        session.execute_query(
            query,
            Path("/tmp/out.parquet"),
            experiment_id=1,
            qp=make_query_params(execution_backend="qep_task_graph"),
        )
        assert (
            call.set_optimization_parameter("execution_backend", "qep_task_graph")
            in cli.method_calls
        )

    @pytest.mark.parametrize(
        ("server_alive", "expected"),
        [
            (True, QueryFailed),
            (False, RestartRequired),
        ],
    )
    def test_cli_error_classified_by_server_liveness(
        self,
        server_alive: bool,
        expected: type[Exception],
    ) -> None:
        session, cli, _ = _session_with_mocks(server_alive=server_alive)
        with_timeout = cli.prepare.return_value.with_timeout
        with_timeout.return_value.with_parquet.return_value.execute.side_effect = GqeCliError(
            ["cmd"], 1, "oops"
        )
        with pytest.raises(expected):
            session.execute_query(
                make_query(name="1", source=QuerySource.SQL, content=b"SELECT 1"),
                Path("/tmp/out.parquet"),
                experiment_id=1,
            )

    @pytest.mark.parametrize(
        ("server_alive", "expected"),
        [
            (True, QueryFailed),
            (False, RestartRequired),
        ],
    )
    def test_parameter_cli_error_classified_by_server_liveness(
        self,
        server_alive: bool,
        expected: type[Exception],
    ) -> None:
        """A failing query-parameter SET is classified like a failing query, not left raw."""
        session, cli, _ = _session_with_mocks(server_alive=server_alive)
        cli.set_optimization_parameter.side_effect = GqeCliError(["cmd"], 1, "connection refused")
        with pytest.raises(expected):
            session.execute_query(
                make_query(name="1", source=QuerySource.SQL, content=b"SELECT 1"),
                Path("/tmp/out.parquet"),
                experiment_id=1,
                qp=make_query_params(),
            )
        cli.prepare.assert_not_called()


class TestGqeSessionReadStorageStats:
    """Tests for GqeSession.read_storage_stats."""

    def test_cli_error_with_live_server_degrades_to_none(self) -> None:
        session, cli, _ = _session_with_mocks(server_alive=True)
        with_timeout = cli.prepare.return_value.with_timeout
        with_timeout.return_value.with_parquet.return_value.execute.side_effect = GqeCliError(
            ["cmd"], 1, "no such table"
        )
        # An engine that does not expose the tables costs the statistics, not the run.
        assert session.read_storage_stats() is None

    def test_cli_error_with_dead_server_raises_restart_required(self) -> None:
        session, cli, _ = _session_with_mocks(server_alive=False)
        with_timeout = cli.prepare.return_value.with_timeout
        with_timeout.return_value.with_parquet.return_value.execute.side_effect = GqeCliError(
            ["cmd"], 1, "connection refused"
        )
        # Continuing here would run every following query against a dead server.
        with pytest.raises(RestartRequired):
            session.read_storage_stats()


class TestSuiteResolveContent:
    """Tests for Suite.resolve_content."""

    def test_sql_with_sql_file_reads_bytes(self, tmp_path: Path) -> None:
        sql_file = tmp_path / "q1.sql"
        sql_file.write_text("SELECT 1;")
        content = TpchSuite.resolve_content(
            "1",
            QuerySource.SQL,
            sql_file,
            1.0,
            "int64",
            load_all_data=True,
            decimal_type="double",
            execution_backend="legacy_task_graph",
        )
        assert content == b"SELECT 1;"

    def test_sql_without_sql_file_generates(self) -> None:
        content = TpchSuite.resolve_content(
            "1",
            QuerySource.SQL,
            None,
            1.0,
            "int64",
            load_all_data=True,
            decimal_type="double",
            execution_backend="legacy_task_graph",
        )
        assert b"SELECT" in content.upper()


class TestGqeSessionLoadData:
    """Tests for GqeSession.load_data."""

    def test_loads_schema_and_tables(self) -> None:
        session, cli, _ = _session_with_mocks()
        session._tables = [("customer", Path("/ds/customer")), ("orders", Path("/ds/orders"))]
        session.load_data()
        cli.load_schema.assert_called_once_with(
            "CREATE TABLE t (id INT) WITH (storage_kind = 'shared_segment_memory');"
        )
        assert cli.load_table.call_count == 2

    def test_drops_tables_before_loading(self) -> None:
        session, cli, _ = _session_with_mocks()
        session._tables = [("customer", Path("/ds/customer"))]
        session.load_data()
        cli.drop_table.assert_called_once_with("customer")

    def test_restores_legacy_backend_before_loading(self) -> None:
        session, cli, _ = _session_with_mocks()
        session.load_data()
        assert cli.method_calls[0] == call.set_optimization_parameter(
            "execution_backend", "legacy_task_graph"
        )

    def test_sends_serialized_dataset_location(self) -> None:
        """The snapshot root goes out on every load; the engine reads it at CREATE TABLE."""
        session, cli, _ = _session_with_mocks()
        session._args.dataset = Path("/ds")
        session.load_data()
        assert (
            call.set_optimization_parameter("serialized_dataset_location", "/ds")
            in cli.method_calls
        )

    def test_cli_error_raises_restart_required(self) -> None:
        session, cli, _ = _session_with_mocks()
        session._tables = [("customer", Path("/ds/customer"))]
        cli.load_schema.side_effect = GqeCliError(["cmd"], 1, "load failed")
        with pytest.raises(RestartRequired):
            session.load_data()

    def test_deserialize_skips_the_copy(self) -> None:
        """CREATE TABLE reconstructs the table from its snapshot, so there is nothing to COPY."""
        session, cli, _ = _session_with_mocks(deserialize_from_disk=True)
        session._tables = [("customer", Path("/ds/customer")), ("orders", Path("/ds/orders"))]
        session.load_data()
        cli.load_schema.assert_called_once_with(
            "CREATE TABLE t (id INT) WITH (storage_kind = 'shared_segment_memory');"
        )
        assert cli.load_table.call_count == 0

    def test_serialize_still_copies(self) -> None:
        """Serialization happens after the COPY, so the COPY must still run."""
        session, cli, _ = _session_with_mocks(serialize_to_disk=True)
        session._tables = [("customer", Path("/ds/customer"))]
        session.load_data()
        assert cli.load_table.call_count == 1


class TestGqeSessionStartProfile:
    """Tests for GqeSession.start_profile."""

    def test_nsys_error_raises_restart_required(self) -> None:
        session, _, _ = _session_with_mocks()
        session._profiler = MagicMock()
        session._profiler.start.side_effect = NsysError(["nsys", "start"], 1, "no such session")
        with pytest.raises(RestartRequired):
            session.start_profile("Q0_e1")


class TestRunGroupResilience:
    """Tests for _run_group failure handling."""

    def _make_group(self, n_queries: int = 2, reference_file: Path | None = None) -> DataLoadGroup:
        return DataLoadGroup(
            data_info=make_data_info(),
            queries=[
                (
                    make_query(name=f"Q{i}", source=QuerySource.SQL, reference_file=reference_file),
                    make_query_params(),
                )
                for i in range(n_queries)
            ],
        )

    def _mock_db_mgr(self) -> MagicMock:
        mgr = MagicMock()
        mgr.__enter__ = MagicMock(return_value=MagicMock())
        mgr.__exit__ = MagicMock(return_value=False)
        return mgr

    def _mock_session(self) -> MagicMock:
        return _mock_gqe_session()

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.promote_run", return_value=0.5)
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_query_failure_continues_to_next_query(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_promote: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        group = self._make_group(n_queries=2)
        session = self._mock_session()
        session.execute_query.side_effect = [QueryFailed("boom"), None]

        _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))
        assert len(group.queries) == 0
        assert session.execute_query.call_count == 2
        # Only the successful second query's run is promoted.
        mock_promote.assert_called_once()

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.promote_run", return_value=0.5)
    @patch("gqe_bench.runner.record_run_failure")
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_restart_required_propagates_with_remaining_queries(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_replace: MagicMock,
        mock_promote: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        group = self._make_group(n_queries=3)
        session = self._mock_session()
        session.execute_query.side_effect = [None, RestartRequired("dead")]

        with pytest.raises(RestartRequired):
            _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))
        assert len(group.queries) == 1

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.promote_run", return_value=0.5)
    @patch("gqe_bench.runner.record_run_failure")
    @patch("gqe_bench.runner.validate_parquet")
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_validation_failure_deletes_and_continues(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_validate: MagicMock,
        mock_replace: MagicMock,
        mock_promote: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        group = self._make_group(n_queries=2, reference_file=Path("/ref/q.parquet"))
        session = self._mock_session()
        mock_validate.side_effect = [ValidationFailed("mismatch"), None]

        _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))
        assert len(group.queries) == 0
        mock_replace.assert_called_once()
        # The failed-validation run is never promoted; the passing one is.
        mock_promote.assert_called_once()

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.record_run_failure")
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_non_session_exception_propagates(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_replace: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        group = self._make_group(n_queries=2)
        session = self._mock_session()
        session.execute_query.side_effect = ValueError("programmer bug")

        with pytest.raises(ValueError, match="programmer bug"):
            _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))
        mock_replace.assert_not_called()


class TestRunGroupValidation:
    """Runner-side validation plumbing (reference_file handling)."""

    def _make_group(self, reference_file: Path | None) -> DataLoadGroup:
        return DataLoadGroup(
            data_info=make_data_info(),
            queries=[
                (
                    make_query(name="Q0", source=QuerySource.SQL, reference_file=reference_file),
                    make_query_params(),
                ),
            ],
        )

    def _mock_db_mgr(self) -> MagicMock:
        mgr = MagicMock()
        mgr.__enter__ = MagicMock(return_value=MagicMock())
        mgr.__exit__ = MagicMock(return_value=False)
        return mgr

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.promote_run", return_value=0.5)
    @patch("gqe_bench.runner.validate_parquet")
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_no_reference_file_skips_validation(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_validate: MagicMock,
        mock_promote: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        group = self._make_group(reference_file=None)
        session = _mock_gqe_session()

        _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))
        mock_validate.assert_not_called()

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.promote_run", return_value=0.5)
    @patch("gqe_bench.runner.validate_parquet")
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_reference_file_triggers_validation(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_validate: MagicMock,
        mock_promote: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        ref = Path("/ref/q.parquet")
        group = self._make_group(reference_file=ref)
        session = _mock_gqe_session()

        _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))
        mock_validate.assert_called_once()
        call_args = mock_validate.call_args
        assert call_args.args[1] == ref


class TestRunGroupStorageStats:
    """Storage statistics collected once per data load and recorded per query_info."""

    def _make_group(self, n_queries: int = 2) -> DataLoadGroup:
        return DataLoadGroup(
            data_info=make_data_info(),
            queries=[
                (make_query(name=f"Q{i}", source=QuerySource.SQL), make_query_params())
                for i in range(n_queries)
            ],
        )

    def _mock_db_mgr(self) -> MagicMock:
        mgr = MagicMock()
        mgr.__enter__ = MagicMock(return_value=MagicMock())
        mgr.__exit__ = MagicMock(return_value=False)
        return mgr

    @patch("gqe_bench.runner.record_storage_stats", return_value=3)
    @patch("gqe_bench.runner.query_info_id_for", side_effect=[7, 8])
    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.promote_run", return_value=0.5)
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_stats_recorded_once_per_query_info(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_promote: MagicMock,
        mock_rec: MagicMock,
        mock_qi: MagicMock,
        mock_stats: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        session = _mock_gqe_session()
        table_stats, column_stats = MagicMock(), MagicMock()
        session.read_storage_stats.return_value = (table_stats, column_stats)

        _run_group(session, self._make_group(), MagicMock(), _make_sweep(self._mock_db_mgr()))

        # The engine is read once per data load, not once per experiment.
        session.read_storage_stats.assert_called_once()
        assert mock_stats.call_count == 2
        assert mock_stats.call_args.args[-2:] == (table_stats, column_stats)

    @patch("gqe_bench.runner.record_storage_stats", return_value=3)
    @patch("gqe_bench.runner.query_info_id_for", return_value=7)
    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.promote_run", return_value=0.5)
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_stats_not_repeated_for_same_query_info(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_promote: MagicMock,
        mock_rec: MagicMock,
        mock_qi: MagicMock,
        mock_stats: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        session = _mock_gqe_session()
        session.read_storage_stats.return_value = (MagicMock(), MagicMock())

        _run_group(session, self._make_group(), MagicMock(), _make_sweep(self._mock_db_mgr()))
        mock_stats.assert_called_once()

    @patch("gqe_bench.runner.record_storage_stats")
    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.promote_run", return_value=0.5)
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_unavailable_stats_skip_recording(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_promote: MagicMock,
        mock_rec: MagicMock,
        mock_stats: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        session = _mock_gqe_session()
        session.read_storage_stats.return_value = None

        _run_group(session, self._make_group(), MagicMock(), _make_sweep(self._mock_db_mgr()))
        mock_stats.assert_not_called()


class TestRunGroupProfiling:
    """Tests that _run_group drives GqeSession.profile correctly.

    One profile per experiment (not per run), labeled
    f"Q{query.name}_{data_info.decimal_type}_e{exp_id}", closed even when the
    repeat loop exits early on failure.
    """

    def _make_group(self, n_queries: int = 2) -> DataLoadGroup:
        # Bare names, like real Query.name values (e.g. "13_fused") -- the
        # QUERY_NAME_PREFIX ("Q") is applied by _run_group, not stored on Query.
        return DataLoadGroup(
            data_info=make_data_info(),
            queries=[
                (make_query(name=str(i), source=QuerySource.SQL), make_query_params())
                for i in range(n_queries)
            ],
        )

    def _mock_db_mgr(self) -> MagicMock:
        mgr = MagicMock()
        mgr.__enter__ = MagicMock(return_value=MagicMock())
        mgr.__exit__ = MagicMock(return_value=False)
        return mgr

    @patch("gqe_bench.runner.record_experiment", side_effect=[1, 2])
    @patch("gqe_bench.runner.promote_run", return_value=0.5)
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_start_profile_called_once_per_experiment_with_label(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_promote: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        group = self._make_group(n_queries=2)
        session = _mock_gqe_session()

        _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))

        assert session.profile.call_args_list == [
            call("Q0_double_e1"),
            call("Q1_double_e2"),
        ]
        assert session.profile.return_value.__exit__.call_count == 2

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.record_run_failure")
    @patch("gqe_bench.runner.next_run_number", return_value=0)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_profile_closed_even_when_restart_required(
        self,
        mock_dims: MagicMock,
        mock_next: MagicMock,
        mock_replace: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        group = self._make_group(n_queries=1)
        session = _mock_gqe_session()
        session.execute_query.side_effect = RestartRequired("dead")

        with pytest.raises(RestartRequired):
            _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))

        session.profile.assert_called_once_with("Q0_double_e1")
        session.profile.return_value.__exit__.assert_called_once()


class TestVerifyReferenceFiles:
    """Pre-flight reference-file check (runner._verify_reference_files)."""

    @staticmethod
    def _group(*reference_files: Path | None) -> DataLoadGroup:
        return DataLoadGroup(
            data_info=make_data_info(),
            queries=[
                (
                    make_query(name=f"Q{i}", source=QuerySource.SQL, reference_file=ref),
                    make_query_params(),
                )
                for i, ref in enumerate(reference_files)
            ],
        )

    def test_missing_reference_raises(self, tmp_path: Path) -> None:
        from gqe_bench.runner import _verify_reference_files

        group = self._group(tmp_path / "absent.parquet")
        with pytest.raises(FileNotFoundError, match=r"absent\.parquet \(missing\)"):
            _verify_reference_files([group])

    def test_empty_reference_raises(self, tmp_path: Path) -> None:
        from gqe_bench.runner import _verify_reference_files

        empty = tmp_path / "empty.parquet"
        empty.touch()
        group = self._group(empty)
        with pytest.raises(FileNotFoundError, match=r"empty\.parquet \(empty\)"):
            _verify_reference_files([group])

    def test_present_nonempty_passes(self, tmp_path: Path) -> None:
        from gqe_bench.runner import _verify_reference_files

        ref = tmp_path / "ok.parquet"
        ref.write_bytes(b"PAR1")
        _verify_reference_files([self._group(ref)])

    def test_all_none_passes(self) -> None:
        from gqe_bench.runner import _verify_reference_files

        _verify_reference_files([self._group(None, None)])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
