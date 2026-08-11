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

"""Pytest configuration shared by unit and integration tests.

Holds the `pytest_addoption` hook (must be in the rootmost conftest on
the test path so options register before argument parsing) and the
`fake_dataset` fixture used by unit tests. Integration-only fixtures
live in `tests/integration/conftest.py`.
"""

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options. Most are consumed by integration fixtures."""
    parser.addoption(
        "--database",
        action="store",
        default=None,
        help="Path to a .db3 SQLite database from a benchmark sweep",
    )
    parser.addoption(
        "--node-manager-bin",
        action="store",
        default=None,
        help="Path to the gqe_node_manager binary",
    )
    parser.addoption(
        "--task-manager-bin",
        action="store",
        default=None,
        help="Path to the gqe_task_manager binary",
    )
    parser.addoption("--cli-bin", action="store", default=None, help="Path to the gqe-cli binary")
    parser.addoption(
        "--ci-config",
        action="store",
        default=None,
        help="Path to a CI json5 config used by the auto-sweep fixture and the "
        "ci_config / ci_db_path fixtures. Fixtures skip if omitted.",
    )


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Temp directory representing a dataset path for unit tests.

    Named sf100_test to emulate TPC-H dataset path format so that
    TpchSuite.infer_scale_factor extracts sf=100 from the path. No
    schema.sql is written — generate_groups consumes the DDL as a string
    parameter (see ``fake_schema_ddl``).
    """
    ds = tmp_path / "sf100_test"
    ds.mkdir()
    return ds


@pytest.fixture
def fake_schema_ddl() -> str:
    """Minimal DDL string for unit tests that exercise generate_groups."""
    return "CREATE TABLE lineitem (l_orderkey BIGINT NOT NULL);"
