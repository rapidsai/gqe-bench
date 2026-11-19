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

"""Fixtures for integration tests.

CLI options (`--ci-config`, `--database`, `--node-manager-bin`, `--task-manager-bin`,
`--cli-bin`) are registered by the parent ``tests/conftest.py``; this conftest
only provides the fixtures that consume them. Binary and plugin paths resolve
through the Artifact API in ``gqe_bench.resources`` / ``gqe_bench._artifacts``;
fixtures skip with a per-reason message if an artifact isn't present.
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import json5
import pytest

from gqe_bench._artifacts import CLI, NODE_MANAGER, PLUGIN, TASK_MANAGER
from gqe_bench.resources import Artifact, ArtifactMissing

_CI_DATASET = Path("/tpch_scratch/datasets/sf0.01/sf0.01_id64")


def _has_ci_dataset() -> bool:
    return _CI_DATASET.exists() and (_CI_DATASET / "lineitem").exists()


def _locate_or_skip(artifact: Artifact, override: str | None) -> Path:
    """Return an artifact path, honoring a CLI-flag override or skipping.

    Override wins; otherwise ``artifact.require()``; skip on
    ``ArtifactMissing`` using the exception's message verbatim.
    """
    if override is not None:
        return Path(override)
    try:
        return artifact.require()
    except ArtifactMissing as e:
        pytest.skip(str(e))


def _ci_sweep_preconditions(ci_config_path: Path | None) -> list[str]:
    """Reasons the auto-sweep fixture can't run, or [] if it can.

    Collects missing-artifact reasons across plugin + three binaries plus
    dataset and CI-config presence, so the skip message lists all gaps
    rather than stopping at the first.
    """
    reasons: list[str] = []
    if ci_config_path is None:
        reasons.append("--ci-config not provided")
    elif not ci_config_path.exists():
        reasons.append(f"CI config missing at {ci_config_path}")
    if not _has_ci_dataset():
        reasons.append(f"CI dataset missing at {_CI_DATASET}")
    for artifact in (PLUGIN, NODE_MANAGER, TASK_MANAGER, CLI):
        try:
            artifact.require()
        except ArtifactMissing as e:
            reasons.append(str(e))
    return reasons


def _ci_config_path(request: pytest.FixtureRequest) -> Path | None:
    """Return the --ci-config Path if provided, else None."""
    raw = request.config.getoption("--ci-config")
    return Path(raw) if raw is not None else None


@pytest.fixture(scope="session")
def _sweep_generated_db_path(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path | None:
    """Generate one fresh sweep .db3 per pytest session, or None if unavailable.

    Used as the fallback path for the `db_path` fixture when --database
    is not provided. Runs `runner.main()` once with the CI json5, so
    downstream DB-validation tests see a real .db3 with the current DDL
    and the plugin's rows.

    Returns None (and the consumer fixture skips) when any precondition
    fails.
    """
    if request.config.getoption("--database") is not None:
        return None
    ci_config_path = _ci_config_path(request)
    if _ci_sweep_preconditions(ci_config_path):
        return None

    tmp_dir = tmp_path_factory.mktemp("sweep_generated_db")
    with open(ci_config_path) as f:
        config = json5.load(f)
    db_path = tmp_dir / "output.db3"
    config["output"] = str(db_path)

    config_path = tmp_dir / "ci_sweep.json5"
    config_path.write_text(json.dumps(config))

    from gqe_bench.runner import main as runner_main

    saved_argv = sys.argv
    try:
        sys.argv = ["runner", "--json", str(config_path)]
        runner_main()
    finally:
        sys.argv = saved_argv

    if not db_path.exists():
        return None
    return db_path


@pytest.fixture(scope="session")
def _pretuned_generated_db_path(
    request: pytest.FixtureRequest,
    _sweep_generated_db_path: Path | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path | None:
    """Generate one pretuned-replay .db3 per pytest session, or None if unavailable.

    Pipeline stage (e) — depends on the sweep db3 (stage (c)) produced by
    `_sweep_generated_db_path`. Runs `runner.main()` with --swept-sqlite
    pointing at the sweep db3, against the CI json5's dataset and
    solution paths. Returns the pretuned db3 path.

    Returns None (and the consumer fixtures skip) when the sweep fixture
    is unavailable or any precondition fails.
    """
    if request.config.getoption("--database") is not None:
        return None
    if _sweep_generated_db_path is None:
        return None
    ci_config_path = _ci_config_path(request)
    if _ci_sweep_preconditions(ci_config_path):
        return None

    with open(ci_config_path) as f:
        config = json5.load(f)

    tmp_dir = tmp_path_factory.mktemp("pretuned_generated_db")
    pretuned_db = tmp_dir / "pretuned.db3"

    from gqe_bench.runner import main as runner_main

    saved_argv = sys.argv
    try:
        sys.argv = [
            "runner",
            "--swept-sqlite",
            str(_sweep_generated_db_path),
            "--dataset",
            str(config["dataset"]),
            "--solution",
            str(config["solution"]),
            "--output",
            str(pretuned_db),
            "--load-all-data",
            "--time-breakdown",
        ]
        runner_main()
    finally:
        sys.argv = saved_argv

    if not pretuned_db.exists():
        return None
    return pretuned_db


@pytest.fixture(scope="module")
def db_path(
    request: pytest.FixtureRequest,
    _sweep_generated_db_path: Path | None,
) -> Path:
    """Sweep .db3 for tests that need a full parameter grid.

    Resolution order:
      1. --database <path> CLI flag (explicit escape hatch, e.g. to
         validate an archived sweep).
      2. A fresh sweep DB auto-generated once per session via the CI
         json5 config. Requires the CI dataset, all four built artifacts,
         and the CI config template.
      3. Skip, with one reason per missing precondition.

    Used by tests whose assertions require the sweep's full param grid
    (e.g. `generate_pretuned_groups` selecting best-of). Tests that should
    validate invariants on both the sweep AND the pretuned db3 use the
    parametrized `experiment_db_path` fixture instead.
    """
    val = request.config.getoption("--database")
    if val is not None:
        return Path(val)
    if _sweep_generated_db_path is not None:
        return _sweep_generated_db_path

    reasons = _ci_sweep_preconditions(_ci_config_path(request))
    pytest.skip("No --database and auto-sweep unavailable: " + "; ".join(reasons))


@pytest.fixture(scope="module", params=["sweep", "pretuned"])
def experiment_db_path(
    request: pytest.FixtureRequest,
    _sweep_generated_db_path: Path | None,
    _pretuned_generated_db_path: Path | None,
) -> Path:
    """Parametrized .db3 — yields the sweep db3 then the pretuned db3.

    Tests using this fixture (transitively, via `db_connection` /
    `db_cursor`) run twice per case: once against the sweep output and
    once against the pretuned-replay output. Asserting schema/view
    invariants on both stages catches regressions specific to one of
    them.

    --database <path> overrides both params with the same path; the
    parametrize still runs twice but consistently against the same db.
    """
    val = request.config.getoption("--database")
    if val is not None:
        return Path(val)

    if request.param == "sweep":
        if _sweep_generated_db_path is None:
            reasons = _ci_sweep_preconditions(_ci_config_path(request))
            pytest.skip("Sweep db3 unavailable: " + "; ".join(reasons))
        return _sweep_generated_db_path
    else:
        if _pretuned_generated_db_path is None:
            reasons = _ci_sweep_preconditions(_ci_config_path(request))
            pytest.skip("Pretuned db3 unavailable: " + "; ".join(reasons))
        return _pretuned_generated_db_path


@pytest.fixture(scope="module")
def db_connection(experiment_db_path: Path) -> sqlite3.Connection:
    """Read-only connection — parametrized over sweep + pretuned db3s."""
    conn = sqlite3.connect(f"file:{experiment_db_path}?mode=ro", uri=True)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def db_cursor(db_connection: sqlite3.Connection) -> sqlite3.Cursor:
    """Cursor for view/table invariant tests, parametrized via db_connection."""
    return db_connection.cursor()


@pytest.fixture(scope="module")
def server_bin(request: pytest.FixtureRequest) -> Path:
    """Node manager binary — from --node-manager-bin flag, or package-resolved."""
    return _locate_or_skip(NODE_MANAGER, request.config.getoption("--node-manager-bin"))


@pytest.fixture(scope="module")
def task_manager_bin(request: pytest.FixtureRequest) -> Path:
    """Task manager binary — from --task-manager-bin flag, or package-resolved."""
    return _locate_or_skip(TASK_MANAGER, request.config.getoption("--task-manager-bin"))


@pytest.fixture(scope="module")
def cli_bin(request: pytest.FixtureRequest) -> Path:
    """CLI binary — from --cli-bin flag, or package-resolved."""
    return _locate_or_skip(CLI, request.config.getoption("--cli-bin"))


@pytest.fixture
def ci_config(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """Load the --ci-config json5 with output redirected to tmp_path.

    Returns (config_dict, written_config_path). Tests can modify the dict
    and re-write before running. Skips when --ci-config is not passed —
    the fixture has no built-in default; integration runs depend on the
    caller supplying a config.
    """
    cfg = _ci_config_path(request)
    if cfg is None:
        pytest.skip("--ci-config not provided")
    if not cfg.exists():
        pytest.skip(f"CI config not found: {cfg}")

    with open(cfg) as f:
        config = json5.load(f)
    config["output"] = str(tmp_path / "output.db3")

    config_path = tmp_path / "test_config.json5"
    config_path.write_text(json.dumps(config))
    return config, config_path


@pytest.fixture
def ci_db_path(ci_config: tuple[dict[str, Any], Path]) -> Path:
    """The output db3 path from the CI config fixture."""
    return Path(ci_config[0]["output"])
