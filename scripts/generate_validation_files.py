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

import argparse
import os
import re
import sys

import duckdb


# TODO: DRY this function
# Importing from gqe.benchmark.utils is not ideal, as it requires building GQE first
def parse_scale_factor(path: str) -> float:
    predicate = re.compile(".*(?:sf|SF)([0-9.]+)([kK]?).*")
    matches = predicate.match(path)

    if matches is None:
        return None

    scale_factor = float(matches.group(1))
    if matches.group(2):
        scale_factor = scale_factor * 1000

    return scale_factor


def delete_database(database_path):
    if os.path.exists(database_path):
        os.remove(database_path)


def tpch_load_sql(dataset):
    # Returns a list of SQL statements to load the TPC-H tables from Parquet
    return [
        "INSTALL parquet;",
        "LOAD parquet;",
        "INSTALL tpch;",
        "LOAD tpch;",
        "-- Create TPC-H schema",
        "CALL dbgen(sf = 0);",
        "-- Load data",
        ".print 'Loading customer'",
        f"INSERT INTO customer SELECT * FROM read_parquet('{os.path.join(dataset, 'customer', '*.parquet')}');",
        ".print 'Loading lineitem'",
        f"INSERT INTO lineitem SELECT * FROM read_parquet('{os.path.join(dataset, 'lineitem', '*.parquet')}');",
        ".print 'Loading nation'",
        f"INSERT INTO nation SELECT * FROM read_parquet('{os.path.join(dataset, 'nation', '*.parquet')}');",
        ".print 'Loading orders'",
        f"INSERT INTO orders SELECT * FROM read_parquet('{os.path.join(dataset, 'orders', '*.parquet')}');",
        ".print 'Loading part'",
        f"INSERT INTO part SELECT * FROM read_parquet('{os.path.join(dataset, 'part', '*.parquet')}');",
        ".print 'Loading partsupp'",
        f"INSERT INTO partsupp SELECT * FROM read_parquet('{os.path.join(dataset, 'partsupp', '*.parquet')}');",
        ".print 'Loading region'",
        f"INSERT INTO region SELECT * FROM read_parquet('{os.path.join(dataset, 'region', '*.parquet')}');",
        ".print 'Loading supplier'",
        f"INSERT INTO supplier SELECT * FROM read_parquet('{os.path.join(dataset, 'supplier', '*.parquet')}');",
    ]


def load_tpch(con, dataset):
    print("Loading data")
    stmts = tpch_load_sql(dataset)
    for stmt in stmts:
        if stmt.startswith(".print"):
            print(stmt.replace(".print", "").strip().strip("'"))
        else:
            con.execute(stmt)


def rewrite_query(query_file, result_file, scale_factor):
    with open(query_file, "r") as f:
        sql = f.read()

    # Q11: Replace FRACTION and sort order
    if re.search(r"[qQ]11\.sql$", query_file):
        q11_fraction = 0.0001 / float(scale_factor)
        # Explicitly convert to decimal notation to avoid parsing issues inside of duckDB with scientific notation defaults
        q11_fraction = format(q11_fraction, ".17f")
        sql = re.sub(r"FRACTION|0\.0001", q11_fraction, sql)
        # Replace ORDER BY value DESC with ORDER BY value DESC, ps_partkey ASC
        sql = re.sub(
            r"(order|ORDER)\s+(by|BY)\s+(value|VALUE)\s+(desc|DESC)",
            "ORDER BY value DESC, ps_partkey ASC",
            sql,
        )
    # Q12: Cast INT128 to INT64 for sum columns
    elif re.search(r"[qQ]12\.sql$", query_file):
        # Replace sum(...) as low_line_count/high_line_count with sum(...)::INT64 as ...
        sql = re.sub(
            r"((sum|SUM)\([^,]+\))\s+((as|AS)\s+(low|high)_line_count)",
            r"\1::INT64 \3",
            sql,
        )

    # Wrap in COPY TO Parquet
    wrapped_sql = (
        f"COPY ({sql.strip().rstrip(';')}) TO '{result_file}' (FORMAT parquet, COMPRESSION snappy);"
    )
    return wrapped_sql


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


def run_queries(con, query_file_paths, results, scale_factor):
    for query_file_path in query_file_paths:
        query_file = os.path.basename(query_file_path)
        print(f"Query {query_file}")

        query_str = os.path.splitext(query_file)[0]
        result_file = os.path.join(results, f"{query_str}.parquet")

        sql = rewrite_query(query_file_path, result_file, scale_factor)
        con.execute(sql)


def main():
    parser = argparse.ArgumentParser(description="DuckDB TPC-H validation script")
    parser.add_argument("dataset", help="Parquet input directory")
    parser.add_argument("queries_sql", help="TPC-H SQL files directory or single SQL file")
    parser.add_argument("results", help="Validation results directory")
    args = parser.parse_args()

    dataset = args.dataset
    queries_sql = args.queries_sql
    results = args.results

    scale_factor = parse_scale_factor(args.dataset)
    duckdb_database = "/tmp/duck.db"

    if not os.path.isdir(results):
        print(f"error: directory {results} doesn't exist")
        sys.exit(1)

    query_file_paths = collect_query_file_paths(queries_sql)

    # Ensure a fresh temporary database in /tmp
    delete_database(duckdb_database)
    con = duckdb.connect(duckdb_database)

    load_tpch(con, dataset)

    run_queries(con, query_file_paths, results, scale_factor)

    con.close()
    delete_database(duckdb_database)


if __name__ == "__main__":
    main()
