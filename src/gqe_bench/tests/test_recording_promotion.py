# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the assumed-fail run lifecycle (promote_run / record_run_failure).

Exercises the Python half against a database built from the real DDL,
simulating the plugin's writes by hand. FKs are not enforced, matching the
production connections; the ``prevent_repeat_run`` /
``prevent_repeat_failed_run`` triggers are active.
"""

import importlib.resources
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from database_benchmarking_tools.experiment import ExperimentConnection
from gqe_bench._artifacts import (
    STAGE_BUILD_METRIC,
    STAGE_COLLECT_METRIC,
    STAGE_EXECUTE_METRIC,
    UNVALIDATED_RUN_MSG,
)
from gqe_bench.recording import (
    _GQE_EXTENSION_DDL,
    next_run_number,
    promote_run,
    record_run_failure,
)

_EXP_ID = 1
_RUN_NUM = 0


@pytest.fixture
def edb(tmp_path: Path) -> Iterator[ExperimentConnection]:
    """An ExperimentConnection on a database built from the real DDL."""
    db_path = tmp_path / "test.db3"
    base_ddl = (
        importlib.resources.files("database_benchmarking_tools.sql")
        .joinpath("create_experiment_db.sql")
        .read_text(encoding="utf-8")
    )
    conn = ExperimentConnection(str(db_path), "testhost")
    conn._cursor.executescript(base_ddl)
    conn._cursor.executescript(_GQE_EXTENSION_DDL)
    conn.commit()
    yield conn


def _simulate_plugin_write(
    edb: ExperimentConnection,
    *,
    build_s: float = 1.25,
    execute_s: float = 2.5,
    collect_s: float = 0.25,
) -> None:
    """Write what run_writer::write_run commits: the pending failed_run row
    and the cross-rank (NULL gpu) stage rows."""
    cur = edb._cursor
    cur.execute(
        "INSERT INTO failed_run (fr_experiment_id, fr_number, fr_error_msg) VALUES (?, ?, ?)",
        (_EXP_ID, _RUN_NUM, UNVALIDATED_RUN_MSG),
    )
    for name, value in (
        (STAGE_BUILD_METRIC, build_s),
        (STAGE_EXECUTE_METRIC, execute_s),
        (STAGE_COLLECT_METRIC, collect_s),
    ):
        cur.execute("INSERT OR IGNORE INTO gqe_metric_info (m_name) VALUES (?)", (name,))
        cur.execute(
            "INSERT INTO gqe_run_ext "
            "(re_experiment_id, re_run_number, re_metric_info_id, re_metric_value, re_gpu_info_id) "
            "VALUES (?, ?, (SELECT m_id FROM gqe_metric_info WHERE m_name = ?), ?, NULL)",
            (_EXP_ID, _RUN_NUM, name, value),
        )
    edb.commit()


def _run_rows(edb: ExperimentConnection) -> list[tuple]:
    edb._cursor.execute("SELECT r_experiment_id, r_number, r_duration_s FROM run")
    return edb._cursor.fetchall()


def _failed_run_rows(edb: ExperimentConnection) -> list[tuple]:
    edb._cursor.execute("SELECT fr_experiment_id, fr_number, fr_error_msg FROM failed_run")
    return edb._cursor.fetchall()


class TestPromoteRun:
    def test_promotes_pending_row_with_recovered_duration(self, edb: ExperimentConnection) -> None:
        _simulate_plugin_write(edb, build_s=1.25, execute_s=2.5, collect_s=99.0)

        # Duration is build + execute; collect is excluded.
        assert promote_run(edb, _EXP_ID, _RUN_NUM) == 3.75
        edb.commit()

        assert _run_rows(edb) == [(_EXP_ID, _RUN_NUM, 3.75)]
        assert _failed_run_rows(edb) == []

    def test_no_pending_row_is_a_noop(self, edb: ExperimentConnection) -> None:
        assert promote_run(edb, _EXP_ID, _RUN_NUM) is None
        assert _run_rows(edb) == []
        assert _failed_run_rows(edb) == []

    def test_missing_stage_rows_leave_pending_row_standing(self, edb: ExperimentConnection) -> None:
        edb._cursor.execute(
            "INSERT INTO failed_run (fr_experiment_id, fr_number, fr_error_msg) VALUES (?, ?, ?)",
            (_EXP_ID, _RUN_NUM, UNVALIDATED_RUN_MSG),
        )
        edb.commit()

        assert promote_run(edb, _EXP_ID, _RUN_NUM) is None
        assert _run_rows(edb) == []
        assert _failed_run_rows(edb) == [(_EXP_ID, _RUN_NUM, UNVALIDATED_RUN_MSG)]

    def test_real_failure_record_is_never_promoted(self, edb: ExperimentConnection) -> None:
        _simulate_plugin_write(edb)
        # Overwrite the sentinel with a real error, as record_run_failure would.
        edb._cursor.execute("UPDATE failed_run SET fr_error_msg = 'validation failed: mismatch'")
        edb.commit()

        assert promote_run(edb, _EXP_ID, _RUN_NUM) is None
        assert _run_rows(edb) == []
        assert len(_failed_run_rows(edb)) == 1

    def test_kill_before_promotion_leaves_pessimistic_record(
        self, edb: ExperimentConnection
    ) -> None:
        """A harness killed between the plugin's commit and promotion must
        leave the run counted as failed. Simulated by simply never calling
        promote_run: the pending row is already durable."""
        _simulate_plugin_write(edb)

        assert _run_rows(edb) == []
        assert _failed_run_rows(edb) == [(_EXP_ID, _RUN_NUM, UNVALIDATED_RUN_MSG)]

    def test_uncommitted_promotion_rolls_back_whole(self, edb: ExperimentConnection) -> None:
        """The swap is atomic: a harness killed after promote_run but before
        commit (simulated by rollback) must restore the pending row and leave
        no run row."""
        _simulate_plugin_write(edb)
        assert promote_run(edb, _EXP_ID, _RUN_NUM) is not None
        edb._conn.rollback()

        assert _run_rows(edb) == []
        assert _failed_run_rows(edb) == [(_EXP_ID, _RUN_NUM, UNVALIDATED_RUN_MSG)]

    def test_promotion_satisfies_repeat_run_triggers(self, edb: ExperimentConnection) -> None:
        """The prevent_repeat_run trigger aborts a run INSERT while a
        failed_run with the same key exists; promotion must therefore
        delete-then-insert. This asserts the whole dance survives the
        real triggers."""
        _simulate_plugin_write(edb)
        assert promote_run(edb, _EXP_ID, _RUN_NUM) is not None
        edb.commit()

        # Re-inserting either specialization for the same key now aborts.
        with pytest.raises(sqlite3.IntegrityError):
            edb._cursor.execute(
                "INSERT INTO failed_run (fr_experiment_id, fr_number, fr_error_msg) "
                "VALUES (?, ?, 'dup')",
                (_EXP_ID, _RUN_NUM),
            )


class TestNextRunNumber:
    def test_fresh_experiment_starts_at_zero(self, edb: ExperimentConnection) -> None:
        assert next_run_number(edb, _EXP_ID) == 0

    def test_continues_past_both_specializations(self, edb: ExperimentConnection) -> None:
        edb._cursor.execute(
            "INSERT INTO run (r_experiment_id, r_number, r_duration_s) VALUES (?, 0, 1.0)",
            (_EXP_ID,),
        )
        edb._cursor.execute(
            "INSERT INTO failed_run (fr_experiment_id, fr_number, fr_error_msg) "
            "VALUES (?, 1, 'boom')",
            (_EXP_ID,),
        )
        edb.commit()

        assert next_run_number(edb, _EXP_ID) == 2
        # Other experiments are unaffected.
        assert next_run_number(edb, _EXP_ID + 1) == 0


class TestRecordRunFailure:
    def test_overrides_pending_row_with_real_error(self, edb: ExperimentConnection) -> None:
        _simulate_plugin_write(edb)

        record_run_failure(edb, _EXP_ID, _RUN_NUM, "query failed: boom")
        edb.commit()

        assert _run_rows(edb) == []
        assert _failed_run_rows(edb) == [(_EXP_ID, _RUN_NUM, "query failed: boom")]
        # The plugin's profiling rows for the attempt are gone.
        edb._cursor.execute("SELECT COUNT(*) FROM gqe_run_ext")
        assert edb._cursor.fetchone()[0] == 0

    def test_works_when_plugin_wrote_nothing(self, edb: ExperimentConnection) -> None:
        record_run_failure(edb, _EXP_ID, _RUN_NUM, "server never started")
        edb.commit()

        assert _failed_run_rows(edb) == [(_EXP_ID, _RUN_NUM, "server never started")]
