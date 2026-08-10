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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gqe_bench.cli import GqeCliError
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
    )


def _session_with_mocks(
    server_alive: bool = True,
) -> tuple[GqeSession, MagicMock, MagicMock]:
    """Build a GqeSession whose server + cli are mocks, bypassing __enter__."""
    session = GqeSession.__new__(GqeSession)
    session._args = MagicMock()
    session._env = {}
    session._schema_ddl = "CREATE TABLE t (id INT);"
    session._tables = []
    session._stack = None
    session._server = MagicMock()
    session._server.is_alive.return_value = server_alive
    session._cli = MagicMock()
    session._agg_shm = None
    return session, session._cli, session._server


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


class TestSuiteResolveContent:
    """Tests for Suite.resolve_content."""

    def test_sql_with_sql_file_reads_bytes(self, tmp_path: Path) -> None:
        sql_file = tmp_path / "q1.sql"
        sql_file.write_text("SELECT 1;")
        content = TpchSuite.resolve_content(
            "1", QuerySource.SQL, sql_file, 1.0, "int64", load_all_data=True, decimal_type="double"
        )
        assert content == b"SELECT 1;"

    def test_sql_without_sql_file_generates(self) -> None:
        content = TpchSuite.resolve_content(
            "1", QuerySource.SQL, None, 1.0, "int64", load_all_data=True, decimal_type="double"
        )
        assert b"SELECT" in content.upper()


class TestGqeSessionLoadData:
    """Tests for GqeSession.load_data."""

    def test_loads_schema_and_tables(self) -> None:
        session, cli, _ = _session_with_mocks()
        session._tables = [("customer", Path("/ds/customer")), ("orders", Path("/ds/orders"))]
        session.load_data()
        cli.load_schema.assert_called_once_with("CREATE TABLE t (id INT);")
        assert cli.load_table.call_count == 2

    def test_drops_tables_before_loading(self) -> None:
        session, cli, _ = _session_with_mocks()
        session._tables = [("customer", Path("/ds/customer"))]
        session.load_data()
        cli.drop_table.assert_called_once_with("customer")

    def test_cli_error_raises_restart_required(self) -> None:
        session, cli, _ = _session_with_mocks()
        session._tables = [("customer", Path("/ds/customer"))]
        cli.load_schema.side_effect = GqeCliError(["cmd"], 1, "load failed")
        with pytest.raises(RestartRequired):
            session.load_data()


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
        return MagicMock(spec=GqeSession)

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_query_failure_continues_to_next_query(
        self,
        mock_dims: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        group = self._make_group(n_queries=2)
        session = self._mock_session()
        session.execute_query.side_effect = [QueryFailed("boom"), None]

        _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))
        assert len(group.queries) == 0
        assert session.execute_query.call_count == 2

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.replace_run_with_failure")
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_restart_required_propagates_with_remaining_queries(
        self,
        mock_dims: MagicMock,
        mock_replace: MagicMock,
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
    @patch("gqe_bench.runner.replace_run_with_failure")
    @patch("gqe_bench.runner.validate_parquet")
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_validation_failure_deletes_and_continues(
        self,
        mock_dims: MagicMock,
        mock_validate: MagicMock,
        mock_replace: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        group = self._make_group(n_queries=2, reference_file=Path("/ref/q.parquet"))
        session = self._mock_session()
        mock_validate.side_effect = [ValidationFailed("mismatch"), None]

        _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))
        assert len(group.queries) == 0
        mock_replace.assert_called_once()

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.replace_run_with_failure")
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_non_session_exception_propagates(
        self,
        mock_dims: MagicMock,
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
    @patch("gqe_bench.runner.validate_parquet")
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_no_reference_file_skips_validation(
        self,
        mock_dims: MagicMock,
        mock_validate: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        group = self._make_group(reference_file=None)
        session = MagicMock(spec=GqeSession)

        _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))
        mock_validate.assert_not_called()

    @patch("gqe_bench.runner.record_experiment", return_value=1)
    @patch("gqe_bench.runner.validate_parquet")
    @patch("gqe_bench.runner.insert_data_dimensions", return_value=DataDimensions(1, 1))
    def test_reference_file_triggers_validation(
        self,
        mock_dims: MagicMock,
        mock_validate: MagicMock,
        mock_rec: MagicMock,
    ) -> None:
        from gqe_bench.runner import _run_group

        ref = Path("/ref/q.parquet")
        group = self._make_group(reference_file=ref)
        session = MagicMock(spec=GqeSession)

        _run_group(session, group, MagicMock(), _make_sweep(self._mock_db_mgr()))
        mock_validate.assert_called_once()
        call_args = mock_validate.call_args
        assert call_args.args[1] == ref


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
