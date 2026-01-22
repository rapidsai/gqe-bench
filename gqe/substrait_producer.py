#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
import subprocess
import sys
import tempfile
from typing import Dict, List

import sqlglot

TPCH_TABLES: List[str] = [
    "part",
    "supplier",
    "partsupp",
    "customer",
    "orders",
    "lineitem",
    "nation",
    "region",
]


def create_tmp_dir():
    return tempfile.mkdtemp(prefix="yaml_files_", dir="/tmp")


def parse_tables(sql_file_path: str) -> List[str]:
    with open(sql_file_path, "r", encoding="utf-8") as f:
        sql_text = f.read()
    statements = sqlglot.parse(sql_text)
    tables = []
    for statement in statements:
        for table in statement.find_all(sqlglot.exp.Table):
            tables.append(table.name.lower())
    return tables


def read_sql_file(sql_file_path: str) -> str:
    with open(sql_file_path, "r", encoding="utf-8") as f:
        return f.read().rstrip("\n")


def build_tables_map(
    data_root: str, query_table_names: List[str], schema_table_names: List[str]
) -> Dict[str, Dict[str, str]]:
    table_to_config: Dict[str, Dict[str, str]] = {}
    # Add only the tables that are present in the query
    for table_name in schema_table_names:
        if table_name in query_table_names:
            table_dir = os.path.join(data_root, table_name)
            table_to_config[table_name] = {"directory": table_dir}
    return table_to_config


def derive_output_path(sql_file_path: str, output_dir: str, output_format: str) -> str:
    """Derive the output binary filename from the SQL filename.

    Example: q1.sql -> q1.bin
    """
    base_name = os.path.basename(sql_file_path)
    stem, extension = os.path.splitext(base_name)
    return os.path.join(output_dir, f"{stem}.{output_format}")


def render_yaml(sql_text: str, output_binary: str, tables_map: Dict[str, Dict[str, str]]) -> str:
    """Render the YAML text manually to control block scalars and indentation.

    Example:
    sql: |
      SELECT * FROM table_name
    output: query_name.bin
    tables:
      table_name:
        directory: /path/to/table_name
    """
    lines: List[str] = []
    lines.append("sql: |")
    for line in sql_text.splitlines():
        lines.append(f"  {line}")
    lines.append(f"output: {output_binary}")
    lines.append("tables:")
    for table_name in tables_map.keys():
        table_cfg = tables_map.get(table_name, {})
        directory = table_cfg.get("directory", "")
        lines.append(f"  {table_name}:")
        lines.append(f"    directory: {directory}")
    lines.append("")
    return "\n".join(lines)


def write_text_file(target_path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created {target_path}")


# TODO: DRY with script/generate_validation_files.py
def collect_query_file_paths(queries_sql):
    query_file_paths = []
    if os.path.isdir(queries_sql):
        query_file_paths = [
            os.path.join(queries_sql, f)
            for f in os.listdir(queries_sql)
            if f.endswith(".sql") and os.path.isfile(os.path.join(queries_sql, f))
        ]
    elif os.path.isfile(queries_sql) and queries_sql.endswith(".sql"):
        query_file_paths = [queries_sql]
    else:
        print(f"error: {queries_sql} is not a valid directory or .sql file")
        sys.exit(1)

    return sorted(query_file_paths, key=lambda p: os.path.basename(p))


def create_substrait_plan(
    sql_file_path: str,
    output_dir: str,
    tmp_dir: str,
    data_dir: str,
    schema_table_names: List[str],
) -> None:
    query_table_names = parse_tables(sql_file_path)
    tables_map = build_tables_map(data_dir, query_table_names, schema_table_names)
    sql_text = read_sql_file(sql_file_path)

    output_binary_path = derive_output_path(sql_file_path, output_dir, "bin")
    yaml_text = render_yaml(
        sql_text=sql_text, output_binary=output_binary_path, tables_map=tables_map
    )

    out_yaml_path = derive_output_path(sql_file_path, tmp_dir, "yaml")
    write_text_file(out_yaml_path, yaml_text)

    subprocess.run(["producer", out_yaml_path], timeout=None)
    print(f"Generated substrait plan binary: {output_binary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Substrait plan binary from a SQL file",
    )
    parser.add_argument(
        "dataset",
        help="Dataset directory containing parquet files",
    )
    parser.add_argument(
        "sql_queries",
        help="Path to input SQL files directory or single SQL file",
    )
    parser.add_argument(
        "output",
        help="Output directory for the generated binaries",
    )
    parser.add_argument(
        "--ddl",
        help="Path to the DDL file, if not specified, TPC-H schema is used",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    sql_queries_path = os.path.abspath(args.sql_queries)
    data_root = os.path.abspath(args.dataset)
    output_dir = os.path.abspath(args.output)
    ddl_file_path = os.path.abspath(args.ddl)

    if ddl_file_path:
        schema_table_names = parse_tables(ddl_file_path)
    else:
        schema_table_names = TPCH_TABLES
    tmp_dir = create_tmp_dir()

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    sql_file_paths = collect_query_file_paths(sql_queries_path)

    for sql_file_path in sql_file_paths:
        create_substrait_plan(
            sql_file_path=sql_file_path,
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            data_dir=data_root,
            schema_table_names=schema_table_names,
        )


if __name__ == "__main__":
    main()
