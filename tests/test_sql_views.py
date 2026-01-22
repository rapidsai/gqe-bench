#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Test SQL views in the experiment database.

These tests validate that the SQL views defined in gqe/benchmark/system_under_test.sql
work correctly by querying them against a SQLite database produced by
run_tpch_parameter_sweep.py.

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
        db_cursor.execute("SELECT COUNT(DISTINCT e_name) FROM experiment")
        expected = db_cursor.fetchone()[0]
        assert expected > 0, "Expected non-zero row count"
        db_cursor.execute("SELECT COUNT(*) FROM gqe_best_parameters")
        actual = db_cursor.fetchone()[0]
        assert (
            actual == expected
        ), f"View 'gqe_best_parameters' has {actual} rows, expected {expected}"

    def test_failed_experiments_empty(self, db_cursor):
        """Test that `failed_experiments` view returns an empty result."""
        db_cursor.execute(
            "SELECT e_name, COUNT(*) as count FROM failed_experiments GROUP BY e_name"
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
            WHERE e_name = (SELECT e_name FROM experiment LIMIT 1)
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
