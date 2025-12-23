#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
