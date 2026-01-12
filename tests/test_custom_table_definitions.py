#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for creating CustomTableDefinitions from a SQL DDL file."""

import gqe.lib


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_inline_primary_key_and_unique_constraints(tmp_path):
    """Inline PRIMARY KEY / UNIQUE on columns are marked unique with the correct datatype."""
    ddl = """
    CREATE TABLE t_unique (
        id INT PRIMARY KEY,
        u VARCHAR UNIQUE,
        x INT NOT NULL
    );
    """
    path = _write(tmp_path, "inline_unique.sql", ddl)
    from gqe.table_definition import CustomTableDefinitions

    td = CustomTableDefinitions(path)
    defs = td.definitions["t_unique"]

    # Check unique property
    assert defs["id"][1] == [gqe.lib.ColumnProperty.unique]
    assert defs["u"][1] == [gqe.lib.ColumnProperty.unique]
    assert len(defs["x"]) == 1  # not unique

    # Check datatypes
    # defs[column][0] is the datatype string
    assert defs["id"][0].type_id() == gqe.lib.TypeId.int32
    assert defs["u"][0].type_id() == gqe.lib.TypeId.string
    assert defs["x"][0].type_id() == gqe.lib.TypeId.int32


def test_table_level_primary_key_and_unique_constraints(tmp_path):
    """Table-level PRIMARY KEY(col) and CONSTRAINT ... UNIQUE(col) are marked unique with the correct datatype."""
    ddl = """
    CREATE TABLE t_pk (
        id INT,
        name VARCHAR,
        PRIMARY KEY (id),
        CONSTRAINT uq_name UNIQUE (name)
    );
    """
    path = _write(tmp_path, "table_level_unique.sql", ddl)
    from gqe.table_definition import CustomTableDefinitions

    td = CustomTableDefinitions(path)
    defs = td.definitions["t_pk"]

    assert defs["id"][1] == [gqe.lib.ColumnProperty.unique]
    assert defs["name"][1] == [gqe.lib.ColumnProperty.unique]

    # Also verify datatypes
    assert defs["id"][0].type_id() == gqe.lib.TypeId.int32
    assert defs["name"][0].type_id() == gqe.lib.TypeId.string


def test_multiple_statements_in_one_file(tmp_path):
    """Multiple CREATE TABLE statements are parsed into separate definitions with the correct datatypes."""
    ddl = """
    CREATE TABLE t1 (id INT PRIMARY KEY, v VARCHAR);
    CREATE TABLE t2 (k BIGINT, x DECIMAL, d DATE);
    """
    path = _write(tmp_path, "multi.sql", ddl)
    from gqe.table_definition import CustomTableDefinitions

    td = CustomTableDefinitions(path)
    assert "t1" in td.definitions
    assert "t2" in td.definitions

    # Also verify query_table_definitions returns ColumnTraits with names and types
    tables = td.query_table_definitions(0)
    t1_cols = {col.name for col in tables["t1"]}
    t2_cols = {col.name for col in tables["t2"]}
    assert t1_cols == {"id", "v"}
    assert t2_cols == {"k", "x", "d"}

    # Check datatypes through query_table_definitions
    t1_types = {col.name: col.data_type.type_id() for col in tables["t1"]}
    t2_types = {col.name: col.data_type.type_id() for col in tables["t2"]}
    assert t1_types == {"id": gqe.lib.TypeId.int32, "v": gqe.lib.TypeId.string}
    assert t2_types == {
        "k": gqe.lib.TypeId.int64,
        "x": gqe.lib.TypeId.float64,
        "d": gqe.lib.TypeId.timestamp_days,
    }
