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

import sqlite3

import pytest

from database_benchmarking_tools.experiment import (
    ExperimentDB,
    FailedRun,
    Run,
)


@pytest.fixture
def edb(tmp_path):
    db_path = str(tmp_path / "test.db")
    edb_config = ExperimentDB(db_path)
    edb_config.create_experiment_db()
    with edb_config as conn:
        yield conn, db_path


def read_count(db_path, table):
    conn = sqlite3.connect(db_path)
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return count


class TestTransaction:
    def test_commit_on_success(self, edb):
        conn, db_path = edb
        with conn.transaction():
            conn.insert_run(Run(experiment_id=1, number=0, duration_s=1.5))
        assert read_count(db_path, "run") == 1

    def test_rollback_on_exception(self, edb):
        conn, db_path = edb
        with pytest.raises(ValueError):
            with conn.transaction():
                conn.insert_run(Run(experiment_id=1, number=0, duration_s=1.5))
                raise ValueError("simulated crash")
        assert read_count(db_path, "run") == 0

    def test_batch_commit(self, edb):
        conn, db_path = edb
        with conn.transaction():
            conn.insert_run(Run(experiment_id=1, number=0, duration_s=1.0))
            conn.insert_run(Run(experiment_id=1, number=1, duration_s=2.0))
            conn.insert_run(Run(experiment_id=1, number=2, duration_s=3.0))
        assert read_count(db_path, "run") == 3

    def test_failed_run(self, edb):
        conn, db_path = edb
        with conn.transaction():
            conn.insert_failed_run(FailedRun(experiment_id=1, number=0, error_msg="test error"))
        assert read_count(db_path, "failed_run") == 1

    def test_no_commit_without_transaction(self, edb):
        conn, db_path = edb
        conn.insert_run(Run(experiment_id=1, number=0, duration_s=1.5))
        assert read_count(db_path, "run") == 0
