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
Test experiment tables in the experiment database.

These tests validate that the tables produced by benchmarking runs
contain correct and consistent data.

Usage:
    pytest tests/test_experiment_tables.py --database /path/to/database.db
"""


class TestTimeBreakdown:
    """Tests for the gqe_run_time_breakdown table."""

    def test_row_count_matches_runs(self, db_cursor):
        """Test that gqe_run_time_breakdown has one row per run."""
        db_cursor.execute("SELECT COUNT(*) FROM run")
        expected = db_cursor.fetchone()[0]
        assert expected > 0, "Expected non-zero run count"
        db_cursor.execute("SELECT COUNT(*) FROM gqe_run_time_breakdown")
        actual = db_cursor.fetchone()[0]
        assert (
            actual == expected
        ), f"Table 'gqe_run_time_breakdown' has {actual} rows, expected {expected} (one per run)"

    def test_breakdown_within_run_duration(self, db_cursor):
        """Test that individual breakdown durations do not exceed the total run duration.

        Note: tb_memcpy_s can be zero when running with zero copy enabled,
        and tb_mem_decompress_s can be zero when running without compression.
        """
        db_cursor.execute("""
            SELECT tb.tb_experiment_id, tb.tb_run_number, r.r_duration_s,
                   tb.tb_in_memory_read_task_s, tb.tb_compute_kernel_s,
                   tb.tb_io_kernel_s, tb.tb_memcpy_s,
                   tb.tb_mem_decompress_s, tb.tb_merged_io_activity_s
            FROM gqe_run_time_breakdown tb
            JOIN run r ON tb.tb_experiment_id = r.r_experiment_id AND tb.tb_run_number = r.r_number
        """)
        columns = [
            "tb_in_memory_read_task_s",
            "tb_compute_kernel_s",
            "tb_io_kernel_s",
            "tb_memcpy_s",
            "tb_mem_decompress_s",
            "tb_merged_io_activity_s",
        ]
        for row in db_cursor.fetchall():
            exp_id, run_num, run_duration = row[0], row[1], row[2]
            # Compute kernels and read tasks must always be strictly positive
            read_task = row[3]
            compute_kernel = row[4]
            assert (
                read_task > 0
            ), f"Run ({exp_id},{run_num}): tb_in_memory_read_task_s={read_task} must be > 0"
            assert (
                compute_kernel > 0
            ), f"Run ({exp_id},{run_num}): tb_compute_kernel_s={compute_kernel} must be > 0"
            for i, col_name in enumerate(columns):
                value = row[i + 3]
                assert value <= run_duration, (
                    f"Run ({exp_id},{run_num}): {col_name}={value:.6f}s exceeds "
                    f"run duration={run_duration:.6f}s"
                )
