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

"""Tests for SQL views in the benchmark database.

Validates SQL views defined in gqe_bench/benchmark/system_under_test.sql
against a .db3 produced by a benchmark run.

Usage:
    pytest gqe_bench/tests/integration/test_experiment_views.py -v --database /path/to/database.db
"""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from gqe_bench._artifacts import (
    STAGE_BUILD_METRIC,
    STAGE_COLLECT_METRIC,
    STAGE_EXECUTE_METRIC,
)
from gqe_bench.query_source import QuerySource
from gqe_bench.suites import get_suite
from gqe_bench.suites.tpch import TpchSuite


class TestExperimentViews:
    """Tests for SQL views in the experiment database."""

    def test_gqe_data_info_row_count(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute("SELECT COUNT(*) FROM gqe_data_info_ext")
        expected = db_cursor.fetchone()[0]
        assert expected > 0
        db_cursor.execute("SELECT COUNT(*) FROM gqe_data_info")
        actual = db_cursor.fetchone()[0]
        assert actual == expected

    def test_gqe_run_parameters_row_count(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute("SELECT COUNT(*) FROM run")
        expected = db_cursor.fetchone()[0]
        assert expected > 0
        db_cursor.execute("SELECT COUNT(*) FROM gqe_run_parameters")
        actual = db_cursor.fetchone()[0]
        assert actual == expected

    def test_gqe_run_all_info_row_count(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute("SELECT COUNT(*) FROM run")
        expected = db_cursor.fetchone()[0]
        assert expected > 0
        db_cursor.execute("SELECT COUNT(*) FROM gqe_run_all_info")
        actual = db_cursor.fetchone()[0]
        assert actual == expected

    def test_gqe_best_parameters_row_count(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute(
            "SELECT COUNT(DISTINCT q_name) FROM experiment "
            "JOIN query_info ON experiment.e_query_info_id = query_info.q_id"
        )
        expected = db_cursor.fetchone()[0]
        assert expected > 0
        db_cursor.execute("SELECT COUNT(*) FROM gqe_best_parameters")
        actual = db_cursor.fetchone()[0]
        assert actual == expected

    def test_gqe_best_parameters_validated_row_count(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute(
            "SELECT COUNT(DISTINCT q_name) FROM experiment "
            "JOIN query_info ON experiment.e_query_info_id = query_info.q_id"
        )
        expected = db_cursor.fetchone()[0]
        assert expected > 0
        db_cursor.execute("SELECT COUNT(*) FROM gqe_best_parameters_validated")
        actual = db_cursor.fetchone()[0]
        assert actual == expected

    def test_gqe_best_parameters_validated_sample_size(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute(
            "SELECT e_sample_size, successful_trials FROM gqe_best_parameters_validated"
        )
        for row in db_cursor:
            assert row[0] == (row[1] + 1)

    def test_gqe_flakey_experiments_sample_size(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute("SELECT e_sample_size, successful_trials FROM gqe_flakey_experiments")
        for i, row in enumerate(db_cursor):
            assert row[0] > row[1] + 1

    def test_failed_experiments_empty(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute(
            "SELECT q_name, COUNT(*) as count FROM failed_experiments GROUP BY q_name"
        )
        failed = db_cursor.fetchall()
        if failed:
            details = "\n".join(f"  - {name}: {count} failed" for name, count in failed)
            raise AssertionError(
                f"Expected no failed experiments, but found {len(failed)}:\n{details}"
            )

    @pytest.mark.parametrize("experiment_db_path", ["sweep"], indirect=True)
    def test_every_configured_query_has_validated_run(
        self, db_cursor: sqlite3.Cursor, ci_config: tuple[dict[str, Any], Path]
    ) -> None:
        """Every TPC-H query the runner is configured to attempt produced
        ≥1 fully-validated experiment.

        Sweep-only: pretuned by construction runs a subset of sweep's
        validated set, so this silent-flakiness regression guard only has
        coverage in the sweep variant.

        "Configured" mirrors the suite's resolution: an empty/missing
        `queries` list expands to the full default set, exactly as the
        runner would. Closes the gap between view-row-count invariants
        (which check consistency *within* the DB) and the runner's
        silent error handling — a query whose every repeat
        validation-fails leaves no row in `gqe_best_parameters_validated`,
        but emits no exception and shows nothing anomalous in the
        row-count tests.
        """
        config, _ = ci_config
        suite = get_suite(config.get("suite_name", TpchSuite.NAME))
        expected: set[str] = set()
        for source in config["query_source"]:
            expected.update(suite.available_queries(QuerySource(source), config.get("queries")))
        db_cursor.execute("SELECT q_name FROM gqe_best_parameters_validated")
        seen = {row[0].lstrip("qQ") for row in db_cursor.fetchall()}
        missing = expected - seen
        assert not missing, f"Configured queries with no validated experiment: {sorted(missing)}"

    def test_gqe_compression_stats(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute("SELECT COUNT(*) FROM gqe_compression_stats")
        actual = db_cursor.fetchone()[0]
        db_cursor.execute("SELECT COUNT(*) FROM gqe_column_stats")
        expected = db_cursor.fetchone()[0]
        assert expected == actual

    def test_gqe_compression_stats_per_table(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute("SELECT COUNT(*) FROM gqe_compression_stats_per_table")
        actual = db_cursor.fetchone()[0]
        db_cursor.execute("SELECT COUNT(*) FROM gqe_table_stats")
        expected = db_cursor.fetchone()[0]
        assert expected == actual

    def test_stage_metrics_in_gqe_metric_info(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute(
            "SELECT m_name FROM gqe_metric_info WHERE m_name IN (?, ?, ?)",
            (STAGE_BUILD_METRIC, STAGE_EXECUTE_METRIC, STAGE_COLLECT_METRIC),
        )
        stage_metrics = {row[0] for row in db_cursor.fetchall()}
        assert STAGE_BUILD_METRIC in stage_metrics
        assert STAGE_EXECUTE_METRIC in stage_metrics

    def test_runs_have_stage_metrics(self, db_cursor: sqlite3.Cursor) -> None:
        db_cursor.execute(
            f"""
            SELECT
                r.r_experiment_id,
                r.r_number,
                r.r_duration_s,
                MAX(CASE WHEN m.m_name = ?
                         THEN re.re_metric_value END) as gen_value,
                MAX(CASE WHEN m.m_name = ?
                         THEN re.re_metric_value END) as exec_value,
                MAX(CASE WHEN m.m_name = ?
                         THEN re.re_metric_value END) as output_value
            FROM run r
            LEFT OUTER JOIN gqe_run_ext re
                ON r.r_experiment_id = re.re_experiment_id AND r.r_number = re.re_run_number
            LEFT OUTER JOIN gqe_metric_info m
                ON re.re_metric_info_id = m.m_id
                AND m.m_name IN (?, ?, ?)
            GROUP BY r.r_experiment_id, r.r_number, r.r_duration_s
        """,
            (
                STAGE_BUILD_METRIC,
                STAGE_EXECUTE_METRIC,
                STAGE_COLLECT_METRIC,
                STAGE_BUILD_METRIC,
                STAGE_EXECUTE_METRIC,
                STAGE_COLLECT_METRIC,
            ),
        )
        runs = db_cursor.fetchall()
        assert len(runs) > 0
        for exp_id, run_number, total_duration, gen_value, exec_value, output_value in runs:
            assert (
                gen_value is not None and gen_value > 0
            ), f"Run ({exp_id},{run_number}): {STAGE_BUILD_METRIC} = {gen_value}"
            assert (
                exec_value is not None and exec_value > 0
            ), f"Run ({exp_id},{run_number}): {STAGE_EXECUTE_METRIC} = {exec_value}"
            stage_sum = gen_value + exec_value
            tolerance = max(0.01 * total_duration, 0.001)
            diff = abs(stage_sum - total_duration)
            assert (
                diff <= tolerance
            ), f"Run ({exp_id},{run_number}): duration={total_duration:.6f}s, stages={stage_sum:.6f}s"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
