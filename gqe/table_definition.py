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

from __future__ import annotations  # Enable forward references for type annotations

from abc import ABC, abstractmethod

import sqlglot

import gqe.lib


def check_identifier_type(type_id: gqe.lib.TypeId) -> bool:
    if type_id in [gqe.lib.TypeId.int32, gqe.lib.TypeId.int64]:
        return True
    raise ValueError(f"Invalid identifier type: {type_id}. Must be int32 or int64.")


class TableDefinitions(ABC):
    @abstractmethod
    def query_table_definitions(self, query_idx: int) -> dict[str, list["gqe.lib.ColumnTraits"]]:
        pass


class TPCHTableDefinitions:
    def __init__(
        self,
        identifier_type: gqe.lib.TypeId = gqe.lib.TypeId.int32,
        use_opt_char_type: bool = True,
    ):
        check_identifier_type(identifier_type)

        self.char_type = gqe.lib.DataType(
            gqe.lib.TypeId.int8 if use_opt_char_type else gqe.lib.TypeId.string
        )
        self.identifier_type = gqe.lib.DataType(identifier_type)
        self.integer_type = gqe.lib.DataType(gqe.lib.TypeId.int32)
        self.decimal_type = gqe.lib.DataType(gqe.lib.TypeId.float64)
        self.string_type = gqe.lib.DataType(gqe.lib.TypeId.string)
        self.date_type = gqe.lib.DataType(gqe.lib.TypeId.timestamp_days)

        self.definitions = {
            "part": {
                "p_partkey": [self.identifier_type, [gqe.lib.ColumnProperty.unique]],
                "p_name": [self.string_type],
                "p_mfgr": [self.string_type],
                "p_brand": [self.string_type],
                "p_type": [self.string_type],
                "p_size": [self.integer_type],
                "p_container": [self.string_type],
                "p_retailprice": [self.decimal_type],
                "p_comment": [self.string_type],
            },
            "supplier": {
                "s_suppkey": [self.identifier_type, [gqe.lib.ColumnProperty.unique]],
                "s_name": [self.string_type],
                "s_address": [self.string_type],
                "s_nationkey": [self.identifier_type],
                "s_phone": [self.string_type],
                "s_acctbal": [self.decimal_type],
                "s_comment": [self.string_type],
            },
            "partsupp": {
                "ps_partkey": [self.identifier_type],
                "ps_suppkey": [self.identifier_type],
                "ps_availqty": [self.integer_type],
                "ps_supplycost": [self.decimal_type],
                "ps_comment": [self.string_type],
            },
            "customer": {
                "c_custkey": [self.identifier_type, [gqe.lib.ColumnProperty.unique]],
                "c_name": [self.string_type],
                "c_address": [self.string_type],
                "c_nationkey": [self.identifier_type],
                "c_phone": [self.string_type],
                "c_acctbal": [self.decimal_type],
                "c_mktsegment": [self.string_type],
                "c_comment": [self.string_type],
            },
            "orders": {
                "o_orderkey": [self.identifier_type, [gqe.lib.ColumnProperty.unique]],
                "o_custkey": [self.identifier_type],
                "o_orderstatus": [self.char_type],
                "o_totalprice": [self.decimal_type],
                "o_orderdate": [self.date_type],
                "o_orderpriority": [self.string_type],
                "o_clerk": [self.string_type],
                "o_shippriority": [self.integer_type],
                "o_comment": [self.string_type],
            },
            "lineitem": {
                "l_orderkey": [self.identifier_type],
                "l_partkey": [self.identifier_type],
                "l_suppkey": [self.identifier_type],
                "l_linenumber": [self.integer_type],
                "l_quantity": [self.decimal_type],
                "l_extendedprice": [self.decimal_type],
                "l_discount": [self.decimal_type],
                "l_tax": [self.decimal_type],
                "l_returnflag": [self.char_type],
                "l_linestatus": [self.char_type],
                "l_shipdate": [self.date_type],
                "l_commitdate": [self.date_type],
                "l_receiptdate": [self.date_type],
                "l_shipinstruct": [self.string_type],
                "l_shipmode": [self.string_type],
                "l_comment": [self.string_type],
            },
            "nation": {
                "n_nationkey": [self.identifier_type, [gqe.lib.ColumnProperty.unique]],
                "n_name": [self.string_type],
                "n_regionkey": [self.identifier_type],
                "n_comment": [self.string_type],
            },
            "region": {
                "r_regionkey": [self.identifier_type, [gqe.lib.ColumnProperty.unique]],
                "r_name": [self.string_type],
                "r_comment": [self.string_type],
            },
        }

    def get_column_types(
        self, tables: dict[str, list[str]]
    ) -> dict[str, list["gqe.lib.ColumnTraits"]]:
        definitions = {}
        for table, columns in tables.items():
            definitions[table] = [
                gqe.lib.ColumnTraits(col, *self.definitions[table][col]) for col in columns
            ]
        return definitions

    def query_table_definitions(self, query_idx: int) -> dict[str, list["gqe.lib.ColumnTraits"]]:
        """Return the tables and and columns (encoded as C++ ColumnTraits) requrired by a query."""

        schema = self.get_schema(query_idx)
        return self.get_column_types(schema)

    def get_schema(self, query_idx: int) -> dict[str, list[str]]:
        """Return column and table names required by a query."""

        # Only required columns for the 22 TPC-H queries
        if query_idx == 0:
            tables = {
                "part": [
                    "p_partkey",
                    "p_name",
                    "p_mfgr",
                    "p_brand",
                    "p_type",
                    "p_size",
                    "p_container",
                ],
                "supplier": [
                    "s_suppkey",
                    "s_name",
                    "s_address",
                    "s_nationkey",
                    "s_phone",
                    "s_acctbal",
                    "s_comment",
                ],
                "partsupp": [
                    "ps_partkey",
                    "ps_suppkey",
                    "ps_availqty",
                    "ps_supplycost",
                ],
                "customer": [
                    "c_custkey",
                    "c_name",
                    "c_address",
                    "c_nationkey",
                    "c_phone",
                    "c_acctbal",
                    "c_mktsegment",
                    "c_comment",
                ],
                "orders": [
                    "o_orderkey",
                    "o_custkey",
                    "o_orderstatus",
                    "o_totalprice",
                    "o_orderdate",
                    "o_orderpriority",
                    "o_shippriority",
                    "o_comment",
                ],
                "lineitem": [
                    "l_orderkey",
                    "l_partkey",
                    "l_suppkey",
                    "l_linenumber",
                    "l_quantity",
                    "l_extendedprice",
                    "l_discount",
                    "l_tax",
                    "l_returnflag",
                    "l_linestatus",
                    "l_shipdate",
                    "l_commitdate",
                    "l_receiptdate",
                    "l_shipinstruct",
                    "l_shipmode",
                ],
                "nation": ["n_nationkey", "n_name", "n_regionkey"],
                "region": ["r_regionkey", "r_name"],
            }
        elif query_idx == 1:
            tables = {
                "lineitem": [
                    "l_returnflag",
                    "l_linestatus",
                    "l_quantity",
                    "l_extendedprice",
                    "l_discount",
                    "l_tax",
                    "l_shipdate",
                ]
            }

        elif query_idx == 2:
            tables = {
                "part": ["p_partkey", "p_size", "p_type", "p_mfgr"],
                "supplier": [
                    "s_suppkey",
                    "s_nationkey",
                    "s_acctbal",
                    "s_name",
                    "s_address",
                    "s_phone",
                    "s_comment",
                ],
                "partsupp": ["ps_suppkey", "ps_partkey", "ps_supplycost"],
                "nation": ["n_name", "n_nationkey", "n_regionkey"],
                "region": ["r_name", "r_regionkey"],
            }
        elif query_idx == 3:
            tables = {
                "customer": ["c_custkey", "c_mktsegment"],
                "orders": ["o_orderkey", "o_orderdate", "o_shippriority", "o_custkey"],
                "lineitem": [
                    "l_orderkey",
                    "l_extendedprice",
                    "l_discount",
                    "l_shipdate",
                ],
            }
        elif query_idx == 4:
            tables = {
                "orders": ["o_orderkey", "o_orderpriority", "o_orderdate"],
                "lineitem": ["l_orderkey", "l_commitdate", "l_receiptdate"],
            }
        elif query_idx == 5:
            tables = {
                "customer": ["c_custkey", "c_nationkey"],
                "orders": ["o_orderkey", "o_custkey", "o_orderdate"],
                "lineitem": [
                    "l_orderkey",
                    "l_extendedprice",
                    "l_discount",
                    "l_suppkey",
                ],
                "supplier": ["s_suppkey", "s_nationkey"],
                "nation": ["n_name", "n_nationkey", "n_regionkey"],
                "region": ["r_regionkey", "r_name"],
            }
        elif query_idx == 6:
            tables = {
                "lineitem": [
                    "l_extendedprice",
                    "l_discount",
                    "l_shipdate",
                    "l_quantity",
                ]
            }
        elif query_idx == 7:
            tables = {
                "supplier": ["s_suppkey", "s_nationkey"],
                "lineitem": [
                    "l_suppkey",
                    "l_orderkey",
                    "l_extendedprice",
                    "l_discount",
                    "l_shipdate",
                ],
                "orders": ["o_orderkey", "o_custkey"],
                "customer": ["c_custkey", "c_nationkey"],
                "nation": ["n_name", "n_nationkey", "n_regionkey"],
            }
        elif query_idx == 8:
            tables = {
                "part": ["p_partkey", "p_type"],
                "supplier": ["s_suppkey", "s_nationkey"],
                "lineitem": [
                    "l_partkey",
                    "l_suppkey",
                    "l_orderkey",
                    "l_extendedprice",
                    "l_discount",
                ],
                "orders": ["o_orderkey", "o_custkey", "o_orderdate"],
                "customer": ["c_custkey", "c_nationkey"],
                "nation": ["n_name", "n_nationkey", "n_regionkey"],
                "region": ["r_name", "r_regionkey"],
            }
        elif query_idx == 9:
            tables = {
                "part": ["p_name", "p_partkey"],
                "lineitem": [
                    "l_partkey",
                    "l_suppkey",
                    "l_orderkey",
                    "l_extendedprice",
                    "l_discount",
                    "l_quantity",
                    "l_shipdate",
                ],
                "orders": ["o_orderkey", "o_orderdate"],
                "supplier": ["s_suppkey", "s_nationkey"],
                "nation": ["n_name", "n_nationkey"],
                "partsupp": ["ps_suppkey", "ps_partkey", "ps_supplycost"],
            }
        elif query_idx == 10:
            tables = {
                "customer": [
                    "c_custkey",
                    "c_name",
                    "c_acctbal",
                    "c_phone",
                    "c_address",
                    "c_comment",
                    "c_nationkey",
                ],
                "orders": ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"],
                "lineitem": [
                    "l_orderkey",
                    "l_returnflag",
                    "l_extendedprice",
                    "l_discount",
                ],
                "nation": ["n_nationkey", "n_name"],
            }
        elif query_idx == 11:
            tables = {
                "partsupp": [
                    "ps_partkey",
                    "ps_suppkey",
                    "ps_supplycost",
                    "ps_availqty",
                ],
                "supplier": ["s_suppkey", "s_nationkey"],
                "nation": ["n_name", "n_nationkey"],
            }
        elif query_idx == 12:
            tables = {
                "orders": ["o_orderkey", "o_orderpriority", "o_orderdate"],
                "lineitem": [
                    "l_orderkey",
                    "l_commitdate",
                    "l_receiptdate",
                    "l_shipdate",
                    "l_shipmode",
                ],
            }
        elif query_idx == 13:
            tables = {
                "customer": ["c_custkey"],
                "orders": ["o_orderkey", "o_custkey", "o_comment"],
            }
        elif query_idx == 14:
            tables = {
                "lineitem": [
                    "l_partkey",
                    "l_extendedprice",
                    "l_discount",
                    "l_shipdate",
                ],
                "part": ["p_partkey", "p_type"],
            }
        elif query_idx == 15:
            tables = {
                "lineitem": [
                    "l_suppkey",
                    "l_extendedprice",
                    "l_discount",
                    "l_shipdate",
                ],
                "supplier": ["s_suppkey", "s_name", "s_address", "s_phone"],
            }
        elif query_idx == 16:
            tables = {
                "part": ["p_brand", "p_type", "p_size", "p_partkey"],
                "partsupp": ["ps_partkey", "ps_suppkey"],
                "supplier": ["s_suppkey", "s_comment"],
            }
        elif query_idx == 17:
            tables = {
                "lineitem": ["l_partkey", "l_extendedprice", "l_quantity"],
                "part": ["p_partkey", "p_brand", "p_container"],
            }
        elif query_idx == 18:
            tables = {
                "customer": ["c_custkey", "c_name"],
                "orders": ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"],
                "lineitem": ["l_orderkey", "l_quantity"],
            }
        elif query_idx == 19:
            tables = {
                "lineitem": [
                    "l_partkey",
                    "l_extendedprice",
                    "l_discount",
                    "l_quantity",
                    "l_shipmode",
                    "l_shipinstruct",
                ],
                "part": ["p_partkey", "p_brand", "p_container", "p_size"],
            }
        elif query_idx == 20:
            tables = {
                "part": ["p_partkey", "p_name"],
                "supplier": ["s_suppkey", "s_nationkey", "s_name", "s_address"],
                "partsupp": ["ps_partkey", "ps_suppkey", "ps_availqty"],
                "nation": ["n_name", "n_nationkey"],
                "lineitem": ["l_partkey", "l_suppkey", "l_quantity", "l_shipdate"],
            }
        elif query_idx == 21:
            tables = {
                "supplier": ["s_suppkey", "s_nationkey", "s_name"],
                "lineitem": [
                    "l_suppkey",
                    "l_orderkey",
                    "l_receiptdate",
                    "l_commitdate",
                ],
                "orders": ["o_orderkey", "o_orderstatus"],
                "nation": ["n_nationkey", "n_name"],
            }
        elif query_idx == 22:
            tables = {
                "customer": ["c_custkey", "c_acctbal", "c_phone", "c_nationkey"],
                "orders": ["o_custkey"],
                "nation": ["n_nationkey"],
            }

        return tables


# TODO: Numeric needs to be handled?
mapping_parse_type_to_gqe_type = {
    "INT": gqe.lib.DataType(gqe.lib.TypeId.int32),
    "BIGINT": gqe.lib.DataType(gqe.lib.TypeId.int64),
    "CHAR": gqe.lib.DataType(gqe.lib.TypeId.string),
    "VARCHAR": gqe.lib.DataType(gqe.lib.TypeId.string),
    "DECIMAL": gqe.lib.DataType(gqe.lib.TypeId.float64),
    "DATE": gqe.lib.DataType(gqe.lib.TypeId.timestamp_days),
}


class CustomTableDefinitions(TableDefinitions):
    def __init__(self, ddl_file_path: str):
        self.definitions = self.parse_table_definitions(ddl_file_path)

    def query_table_definitions(self, query_idx: int) -> dict[str, list["gqe.lib.ColumnTraits"]]:
        return {
            table: [gqe.lib.ColumnTraits(col, *type_traits) for col, type_traits in cols.items()]
            for table, cols in self.definitions.items()
        }

    def get_single_column_identifiers(
        self, expression: sqlglot.exp.Expression, column_names: list[str]
    ):
        column_identifiers = expression.find_all(sqlglot.exp.Identifier)
        column_names_cur = [col.name.lower() for col in column_identifiers]
        # Composite column uniqueness (ex. UNIQUE (a,b) ) is not supported by GQE yet.
        # So, we only add the column name if it is a single column.
        if len(column_names_cur) == 1:
            column_names.append(column_names_cur[0])

    def parse_table_definitions(self, ddl_file_path: str) -> dict[str, dict[str, str]]:
        # Parse the DDL file contents into statements (safe: use file contents, not path)
        with open(ddl_file_path, "r", encoding="utf-8") as f:
            ddl_text = f.read()
        statements = sqlglot.parse(ddl_text)
        table_definitions = {}
        for statement in statements:
            # Assume there should be only one table per statement
            table_name = statement.find(sqlglot.exp.Table).name.lower()

            column_names = []
            column_types = []
            unique_column_names = []

            # This parses all column level options
            # [column name] [data type] {NULL | NOT NULL} {column options}
            for column in statement.find_all(sqlglot.exp.ColumnDef):
                column_names.append(column.name.lower())
                column_types.append(
                    mapping_parse_type_to_gqe_type[column.args.get("kind").this.name.upper()]
                )

                for constraint in column.constraints:
                    if str(constraint).upper() == "UNIQUE":
                        unique_column_names.append(column.name.lower())
                    if str(constraint).upper() == "PRIMARY KEY":
                        unique_column_names.append(column.name.lower())

            # This parses the primary key constraint
            #  PRIMARY KEY (column name [, column name ...])
            #  CONSTRAINT [constraint name] PRIMARY KEY (column name [, column name ...])
            for primary_key_expression in statement.find_all(sqlglot.exp.PrimaryKey):
                self.get_single_column_identifiers(primary_key_expression, unique_column_names)

            # This parses the unique column constraint
            #  CONSTRAINT [constraint name] UNIQUE (column name [, column name ...])
            for unique_key_expression in statement.find_all(sqlglot.exp.UniqueColumnConstraint):
                self.get_single_column_identifiers(unique_key_expression, unique_column_names)

            # If the column is in the unique column names, we attach the unique column property to table definition
            table_definitions[table_name] = {}
            for column in column_names:
                if column in unique_column_names:
                    table_definitions[table_name][column] = [
                        column_types[column_names.index(column)],
                        [gqe.lib.ColumnProperty.unique],
                    ]
                else:
                    table_definitions[table_name][column] = [
                        column_types[column_names.index(column)]
                    ]

        return table_definitions
