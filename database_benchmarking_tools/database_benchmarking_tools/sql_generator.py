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
from dataclasses import asdict, fields


def _get_insert_body(cursor: sqlite3.Cursor, entry):
    sql = (
        ",\n".join([f"{ entry._table_prefix }{ field.name }" for field in fields(entry)])
        + "\n) "
        + "VALUES (\n"
        + ",\n".join([f":{ field.name }" for field in fields(entry)])
        + "\n)"
    )
    return sql


def insert(cursor: sqlite3.Cursor, entry):
    sql = (
        f"INSERT INTO { entry._table_name } (\n"
        + _get_insert_body(cursor, entry)
        + f"\nRETURNING { entry._table_prefix }id"
    )
    cursor.execute(sql, asdict(entry))
    return cursor.fetchone()[0]


def insert_natural_key(cursor: sqlite3.Cursor, entry):
    # In cases where the table has a natural key, no need to
    # return autoincrement ID.
    sql = f"INSERT INTO { entry._table_name } (\n" + _get_insert_body(cursor, entry)
    cursor.execute(sql, asdict(entry))


def insert_or_ignore(cursor: sqlite3.Cursor, entry):
    # There is a UNIQUE constraint, so we IGNORE if the row already exists.
    #
    # FIXME: UNIQUE constraints consider NULL values to be distinct. This leads
    # to duplicate entries. Duplicates should be avoided, as we consider NULL
    # values to be equal.
    sql = f"INSERT OR IGNORE INTO { entry._table_name } (\n" + _get_insert_body(cursor, entry)

    cursor.execute(sql, asdict(entry))


def select_id(cursor: sqlite3.Cursor, entry):
    # Retrieve the primary key of the row.
    #
    # We can't use `lastrowid`, because the INSERT OR IGNORE might not insert a
    # new row. In this case, `lastrowid` doesn't contain to the row we want.
    #
    # `IS NULL OR` is necessary to match rows that don't have a NOT NULL constraint.
    sql = (
        f"SELECT { entry._table_prefix }id\n"
        + f"FROM { entry._table_name }\n"
        + "WHERE "
        + "\nAND ".join(
            [
                f"({ entry._table_prefix }{ field.name } IS NULL OR { entry._table_prefix }{ field.name } = :{ field.name })"
                for field in fields(entry)
            ]
        )
    )

    cursor.execute(sql, asdict(entry))
    result = cursor.fetchone()[0]

    # # FIXME: Assert that at most one row is returned by the SELECT query.
    # #
    # # `IS NULL OR` isn't entirely correct, because theoretically it might
    # # return multiple rows. This would occur when searching for ('x', NULL) in a table containing the rows [('x', NULL), ('x', 'y')].
    # next = cursor.fetchone()
    # assert (
    #     next is None
    # ), f"Expected a single result row. Check that columns consist of either NULLs or non-NULLs, but not a mix. Got { entry._table_prefix }id: { result } and { next[0] }"

    return result
