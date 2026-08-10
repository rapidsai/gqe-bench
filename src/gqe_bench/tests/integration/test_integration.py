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
Integration tests for server + CLI together and runner.main() end-to-end.

Uses config_templates/tpch_sweep_CI.json5 — the same config that CI uses.
Requires the CI dataset to be available at /tpch_scratch/datasets/sf0.01/.

Usage:
    pytest gqe_bench/tests/test_integration.py -v
"""

import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from gqe_bench.cli import GqeCli
from gqe_bench.query_source import QuerySource
from gqe_bench.server import GqeServer
from gqe_bench.tests.integration.conftest import _CI_DATASET, _has_ci_dataset


def _run_main(config_path: Path) -> None:
    """Run runner.main() with the given config, restoring sys.argv after."""
    from gqe_bench.runner import main as runner_main

    saved_argv = sys.argv
    try:
        sys.argv = ["runner", "--json", str(config_path)]
        runner_main()
    finally:
        sys.argv = saved_argv


class TestServerCliIntegration:
    """Server + CLI connectivity test with a real table."""

    def test_load_and_query_table(
        self,
        server_bin: Path,
        task_manager_bin: Path,
        cli_bin: Path,
    ) -> None:
        if not _has_ci_dataset():
            pytest.skip(f"Dataset not found: {_CI_DATASET}")

        with GqeServer(server_bin, task_manager_bin) as srv:
            cli = GqeCli(cli_bin, f"http://localhost:{srv.addr.split(':')[-1]}")
            schema_file = _CI_DATASET / "schema.sql"
            cli.load_schema(schema_file.read_text(encoding="utf-8"))
            cli.load_table("nation", _CI_DATASET / "nation")
            cli.prepare(
                QuerySource.SQL, "SELECT n_nationkey, n_name FROM nation LIMIT 5;"
            ).execute()


class TestRunnerMain:
    """End-to-end tests through runner.main()."""

    def test_sweep_all_queries(
        self,
        ci_config: tuple[dict[str, Any], Path],
    ) -> None:
        """All TPC-H queries in the CI config execute without error.

        Behavioural check only — runner.main returning is the success
        signal. DB-state assertions (per-query validation coverage,
        no-failed-experiments, view consistency) live in
        `test_experiment_views.py`, where they run against both the
        sweep and pretuned outputs.
        """
        if not _has_ci_dataset():
            pytest.skip(f"Dataset not found: {_CI_DATASET}")
        _, config_path = ci_config
        _run_main(config_path)

    def test_run_number_contiguous(
        self,
        ci_config: tuple[dict[str, Any], Path],
        ci_db_path: Path,
    ) -> None:
        """Per experiment, run.r_number ∪ failed_run.fr_number = 0..repeat-1.

        Python's `_run_group` iterates `for run_num in range(repeat)` and each
        iteration lands in exactly one of the two tables (plugin-written `run`
        on success, Python-written `failed_run` on validation failure). The
        plugin derives its `r_number` from `COALESCE(MAX(n), -1) + 1` over
        `run ∪ failed_run`, so the sequence is contiguous only because:
          - experiments start with zero prior rows,
          - iteration is sequential,
          - Python is the only other writer,
          - the mutual-exclusion triggers enforce (exp_id, n) appears in
            exactly one table.

        This test pins that invariant. A pre-seeded row, a retry, or
        per-query parallelism would break the contiguity silently.
        """
        if not _has_ci_dataset():
            pytest.skip(f"Dataset not found: {_CI_DATASET}")

        _, config_path = ci_config
        _run_main(config_path)

        assert ci_db_path.exists()
        conn = sqlite3.connect(str(ci_db_path))
        try:
            exp_ids = [row[0] for row in conn.execute("SELECT e_id FROM experiment")]
            for exp_id in exp_ids:
                numbers = [
                    row[0]
                    for row in conn.execute(
                        "SELECT n FROM "
                        "(SELECT r_experiment_id AS e, r_number AS n FROM run "
                        " UNION ALL "
                        " SELECT fr_experiment_id AS e, fr_number AS n FROM failed_run) "
                        "WHERE e = ? ORDER BY n",
                        (exp_id,),
                    )
                ]
                assert numbers == list(
                    range(len(numbers))
                ), f"experiment {exp_id}: run numbers not contiguous 0..N-1, got {numbers}"
        finally:
            conn.close()

    def test_pretuned_mode(
        self,
        tmp_path: Path,
        ci_config: tuple[dict[str, Any], Path],
        server_bin: Path,
        task_manager_bin: Path,
        cli_bin: Path,
    ) -> None:
        """Pretuned mode: sweep produces db3, pretuned reads it."""
        if not _has_ci_dataset():
            pytest.skip(f"Dataset not found: {_CI_DATASET}")

        from gqe_bench.runner import main as runner_main

        # Run sweep to produce a db3
        _, sweep_config_path = ci_config
        _run_main(sweep_config_path)

        sweep_db = Path(ci_config[0]["output"])
        assert sweep_db.exists()

        # Run pretuned from that db3
        pretuned_db = tmp_path / "pretuned.db3"
        saved_argv = sys.argv
        try:
            sys.argv = [
                "runner",
                "--swept-sqlite",
                str(sweep_db),
                "--server-bin",
                str(server_bin),
                "--task-manager-bin",
                str(task_manager_bin),
                "--cli-bin",
                str(cli_bin),
                "--dataset",
                str(_CI_DATASET),
                "--output",
                str(pretuned_db),
                "--load-all-data",  # CI only uses one DataInfo for pretuned
            ]
            runner_main()
        finally:
            sys.argv = saved_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
