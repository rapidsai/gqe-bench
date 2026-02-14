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
Test SQL views in the experiment database.

These tests validate that the SQL views defined in gqe/benchmark/system_under_test.sql
work correctly by querying them against a SQLite database produced by
run_tpch_sweep.py.

Usage:
    pytest tests/test_sql_views.py --database /path/to/database.db
"""


class TestSqlViews:
    """Tests for SQL views in the experiment database."""

    def test_gqe_data_info_row_count(self, db_cursor):
        """Test that `gqe_data_info` view has expected row count."""
        # `gqe_data_info_ext` is a weak entity of `data_info` (many ext per data_info).
        # The view joins them, so row count equals `gqe_data_info_ext` count.
        db_cursor.execute("SELECT COUNT(*) FROM gqe_data_info_ext")
        expected = db_cursor.fetchone()[0]
        assert expected > 0, "Expected non-zero row count"
        db_cursor.execute("SELECT COUNT(*) FROM gqe_data_info")
        actual = db_cursor.fetchone()[0]
        assert actual == expected, f"View 'gqe_data_info' has {actual} rows, expected {expected}"

    def test_gqe_run_parameters_row_count(self, db_cursor):
        """Test that `gqe_run_parameters` view has expected row count."""
        # `run` is a weak entity of `experiment` (many runs per experiment).
        # The view joins `run` with `experiment` and parameters, so row count equals `run` count.
        db_cursor.execute("SELECT COUNT(*) FROM run")
        expected = db_cursor.fetchone()[0]
        assert expected > 0, "Expected non-zero row count"
        db_cursor.execute("SELECT COUNT(*) FROM gqe_run_parameters")
        actual = db_cursor.fetchone()[0]
        assert (
            actual == expected
        ), f"View 'gqe_run_parameters' has {actual} rows, expected {expected}"

    def test_gqe_run_all_info_row_count(self, db_cursor):
        """Test that `gqe_run_all_info` view has expected row count."""
        # `run` is a weak entity of `experiment` (many runs per experiment).
        # The view joins `run` with all info tables, so row count equals `run` count.
        db_cursor.execute("SELECT COUNT(*) FROM run")
        expected = db_cursor.fetchone()[0]
        assert expected > 0, "Expected non-zero row count"
        db_cursor.execute("SELECT COUNT(*) FROM gqe_run_all_info")
        actual = db_cursor.fetchone()[0]
        assert actual == expected, f"View 'gqe_run_all_info' has {actual} rows, expected {expected}"

    def test_gqe_best_parameters_row_count(self, db_cursor):
        """Test that `gqe_best_parameters` view has expected row count."""
        # `gqe_best_parameters` has one row per unique query name, and aggregates
        # the results of the runs. Thus, the row count equals the number of unique
        # query names.
        db_cursor.execute(
            "SELECT COUNT(DISTINCT q_name) FROM experiment JOIN query_info ON experiment.e_query_info_id = query_info.q_id"
        )
        expected = db_cursor.fetchone()[0]
        assert expected > 0, "Expected non-zero row count"
        db_cursor.execute("SELECT COUNT(*) FROM gqe_best_parameters")
        actual = db_cursor.fetchone()[0]
        assert (
            actual == expected
        ), f"View 'gqe_best_parameters' has {actual} rows, expected {expected}"

    def test_gqe_best_parameters_validated_row_count(self, db_cursor):
        """Test that `gqe_best_parameters_validated` view has expected row count."""
        # `gqe_best_parameters` has one row per unique query name, and aggregates
        # the results of the runs. Thus, the row count equals the number of unique
        # query names.
        db_cursor.execute(
            "SELECT COUNT(DISTINCT q_name) FROM experiment JOIN query_info ON experiment.e_query_info_id = query_info.q_id"
        )
        expected = db_cursor.fetchone()[0]
        assert expected > 0, "Expected non-zero row count"
        db_cursor.execute("SELECT COUNT(*) FROM gqe_best_parameters_validated")
        actual = db_cursor.fetchone()[0]
        assert (
            actual == expected
        ), f"View 'gqe_best_parameters_validated' has {actual} rows, expected {expected}"

    def test_gqe_best_parameters_validated_sample_size(self, db_cursor):
        """Test that `gqe_best_parameters_validated` view has expected sample_size property."""
        # `gqe_best_parameters_validated` expects that successful_trials + 1 matches e_sample_size
        # for all rows
        db_cursor.execute(
            "SELECT e_sample_size, successful_trials FROM gqe_best_parameters_validated"
        )
        for row in db_cursor:
            assert (
                row[0] == (row[1] + 1)
            ), f"View 'gqe_best_parameters_validated' has {row[1]} successful_trials, expected {row[0]-1}"

    def test_gqe_flakey_experiments_sample_size(self, db_cursor):
        """Test that `gqe_flakey_experiments` view has expected sample_size property."""
        # `gqe_flakey_experiments` expects all rows to have e_sample_size > successful_trials+1
        db_cursor.execute("SELECT e_sample_size, successful_trials FROM gqe_flakey_experiments")
        for i, row in enumerate(db_cursor):
            print(f"row: {row}")
            assert (
                row[0] > row[1] + 1
            ), f"View 'gqe_flakey_experiments' row {i} has {row[1]} successful_trials, expected {row[0]}"

    def test_failed_experiments_empty(self, db_cursor):
        """Test that `failed_experiments` view returns an empty result."""
        db_cursor.execute(
            "SELECT q_name, COUNT(*) as count FROM failed_experiments GROUP BY q_name"
        )
        failed_experiments = db_cursor.fetchall()
        if failed_experiments:
            details = "\n".join(f"  - {name}: {count} failed" for name, count in failed_experiments)
            raise AssertionError(
                f"Expected no failed experiments, but found {len(failed_experiments)} "
                f"failed experiments:\n{details}"
            )

    def test_gqe_compression_stats(self, db_cursor):
        """Test that `gqe_compression_stats` view has expected row count compared to `gqe_column_stats` because each column should have a compression stat."""
        db_cursor.execute("SELECT COUNT(*) FROM gqe_compression_stats")
        actual = db_cursor.fetchone()[0]
        db_cursor.execute("SELECT COUNT(*) FROM gqe_column_stats")
        expected = db_cursor.fetchone()[0]
        assert (
            expected == actual
        ), f"View 'gqe_compression_stats' has {actual} rows, expected {expected}"

    def test_gqe_compression_stats_per_table(self, db_cursor):
        """Test that `gqe_compression_stats_per_table` view has expected row count compared to `gqe_table_stats` because each table should have a compression stat."""
        db_cursor.execute("SELECT COUNT(*) FROM gqe_compression_stats_per_table")
        actual = db_cursor.fetchone()[0]
        db_cursor.execute("SELECT COUNT(*) FROM gqe_table_stats")
        expected = db_cursor.fetchone()[0]
        assert (
            expected == actual
        ), f"View 'gqe_compression_stats_per_table' has {actual} rows, expected {expected}"

    def test_gqe_compression_stats_per_data_info(self, db_cursor):
        """Test that `gqe_compression_stats_per_data_info` view has expected row count.

        For parameter sweep: each query runs on all data_info configs, so count = num_queries * num_data_info.
        For pretuned: each query runs on one data_info config, so count = num_queries.
        """
        db_cursor.execute("SELECT COUNT(*) FROM gqe_compression_stats_per_data_info")
        actual = db_cursor.fetchone()[0]

        # Check if queries run on multiple data_info configs, pretuned mode would only run on one best data info config.
        db_cursor.execute("""
            SELECT COUNT(DISTINCT e_data_info_ext_id)
            FROM experiment
            WHERE e_query_info_id = (SELECT e_query_info_id FROM experiment LIMIT 1)
        """)
        data_info_per_query = db_cursor.fetchone()[0]

        db_cursor.execute("SELECT COUNT(*) FROM gqe_best_parameters")
        num_queries = db_cursor.fetchone()[0]

        if data_info_per_query > 1:
            # Parameter sweep mode: each query runs on all data_info configs
            db_cursor.execute("SELECT COUNT(*) FROM gqe_data_info")
            num_gqe_data_info = db_cursor.fetchone()[0]
            expected = num_gqe_data_info * num_queries
        else:
            # Pretuned mode: each query runs on one data_info config
            expected = num_queries

        assert (
            expected == actual
        ), f"View 'gqe_compression_stats_per_data_info' has {actual} rows, expected {expected}"

    def test_stage_metrics_in_gqe_metric_info(self, db_cursor):
        """Test that stage timing metrics exist in gqe_metric_info table."""
        # Query for expected stage metric names
        db_cursor.execute("""
            SELECT m_name FROM gqe_metric_info
            WHERE m_name IN ('task_graph_generation', 'task_graph_execution', 'output_generation')
        """)
        stage_metrics = {row[0] for row in db_cursor.fetchall()}

        # Task graph generation and execution are always present
        assert (
            "task_graph_generation" in stage_metrics
        ), "Expected 'task_graph_generation' metric in gqe_metric_info"
        assert (
            "task_graph_execution" in stage_metrics
        ), "Expected 'task_graph_execution' metric in gqe_metric_info"
        # Don't test the presence of stage 'output_generation' because it is not generated when no output is written

    def test_runs_have_stage_metrics(self, db_cursor):
        """Test that each run has associated stage timing metrics and durations sum correctly."""
        # Get stage metrics and total runtime duration for all runs in the database.
        # TODO: The MAX assumes that there is a single stage for each run for each metric.
        # AFAIK, this is not enforced because the GqeRunExt dataclass does not generate a primary key.
        db_cursor.execute("""
            SELECT
                r.r_experiment_id,
                r.r_number,
                r.r_duration_s,
                MAX(CASE WHEN m.m_name = 'task_graph_generation'
                         THEN re.re_metric_value END) as gen_value,
                MAX(CASE WHEN m.m_name = 'task_graph_execution'
                         THEN re.re_metric_value END) as exec_value,
                MAX(CASE WHEN m.m_name = 'output_generation'
                         THEN re.re_metric_value END) as output_value
            FROM run r
            LEFT OUTER JOIN gqe_run_ext re ON r.r_experiment_id = re.re_experiment_id and r.r_number = re.re_run_number
            LEFT OUTER JOIN gqe_metric_info m ON re.re_metric_info_id = m.m_id
                AND m.m_name IN ('task_graph_generation', 'task_graph_execution', 'output_generation')
            GROUP BY r.r_experiment_id, r.r_number, r.r_duration_s
        """)
        runs = db_cursor.fetchall()

        assert len(runs) > 0, "Expected at least one run in database"

        for exp_id, run_number, total_duration, gen_value, exec_value, output_value in runs:
            # Check that stage durations exist and are > 0
            assert (
                gen_value is not None and gen_value > 0
            ), f"Run ({exp_id},{run_number}): 'task_graph_generation' = {gen_value} (must be > 0)"
            assert (
                exec_value is not None and exec_value > 0
            ), f"Run ({exp_id},{run_number}): 'task_graph_execution' = {exec_value} (must be > 0)"

            # Check that sum of stage durations matches total duration
            stage_sum = gen_value + exec_value
            tolerance = max(0.01 * total_duration, 0.001)  # At least 1ms tolerance
            diff = abs(stage_sum - total_duration)
            assert (
                diff <= tolerance
            ), f"Run ({exp_id},{run_number}): total_duration={total_duration:.6f}s, stage_sum={stage_sum:.6f}s (diff={diff:.6f}s)"
