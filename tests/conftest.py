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

"""Pytest configuration and shared fixtures."""

import sqlite3

import pytest


def pytest_addoption(parser):
    """Add command-line options for tests."""
    parser.addoption(
        "--database",
        action="store",
        default=None,
        help="Path to the SQLite database file for SQL view tests",
    )


@pytest.fixture(scope="module")
def database_path(request):
    """Get the database path from command-line option."""
    db_path = request.config.getoption("--database")
    if db_path is None:
        pytest.skip("No database path provided. Use --database option.")
    return db_path


@pytest.fixture(scope="module")
def db_connection(database_path):
    """Create a read-only database connection for the test module."""
    conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def db_cursor(db_connection):
    """Create a database cursor for the test module."""
    return db_connection.cursor()
