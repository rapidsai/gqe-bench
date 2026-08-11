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
Tests for the benchmark suite system.

Usage:
    pytest gqe_bench/tests/test_suites.py -v
"""

import sqlite3
from pathlib import Path

import pytest

from gqe_bench.query_source import QuerySource
from gqe_bench.suites import get_suite
from gqe_bench.suites.base import NoBuiltInDDL, Suite
from gqe_bench.suites.tpch import TpchSuite


class TestSuiteRegistry:
    """Tests for suite registration and lookup."""

    def test_tpch_registered(self) -> None:
        assert get_suite(TpchSuite.NAME) is TpchSuite

    def test_unknown_falls_back(self) -> None:
        assert get_suite("TPC-DS") is Suite


class TestTpchSuite:
    """Tests for the TPC-H suite."""

    def test_sql_queries_count(self) -> None:
        assert len(get_suite(TpchSuite.NAME).available_queries(QuerySource.SQL, None)) == 22

    def test_handcoded_queries_count(self) -> None:
        assert len(get_suite(TpchSuite.NAME).available_queries(QuerySource.HANDCODED, None)) == 25

    def test_query_filter(self) -> None:
        qs = get_suite(TpchSuite.NAME).available_queries(QuerySource.SQL, ["1", "5", "10"])
        assert qs == ["1", "5", "10"]

    def test_tables(self) -> None:
        tables = get_suite(TpchSuite.NAME).tables(Path("/ds"))
        assert len(tables) == 8
        assert tables[0] == ("customer", Path("/ds/customer"))

    def test_schema_default_emits_builtin_ddl(self) -> None:
        ddl = get_suite(TpchSuite.NAME).schema(None)
        assert "CREATE TABLE lineitem" in ddl
        assert "TINYINT" in ddl  # char-column workaround
        assert "BIGINT" in ddl  # default identifier_type

    def test_schema_override_returns_file_content(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.sql"
        body = "CREATE TABLE t (id INT);"
        custom.write_text(body)
        assert get_suite(TpchSuite.NAME).schema(custom) == body

    def test_to_ddl_round_trip(self) -> None:
        suite = get_suite(TpchSuite.NAME)
        ddl = suite.to_ddl()
        props = suite.read_dataset_properties(ddl)
        assert props["char_type"] == "char"
        assert props["identifier_type"] == "int64"
        assert props["not_null"] is False

    def test_to_ddl_emits_primary_keys(self) -> None:
        ddl = get_suite(TpchSuite.NAME).to_ddl()
        expected_pks = {
            "part": "p_partkey",
            "supplier": "s_suppkey",
            "customer": "c_custkey",
            "orders": "o_orderkey",
            "nation": "n_nationkey",
            "region": "r_regionkey",
        }
        for col in expected_pks.values():
            assert f"PRIMARY KEY ({col})" in ddl, f"missing PRIMARY KEY ({col})"

        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(ddl)
            for table, pk_col in expected_pks.items():
                rows = list(conn.execute(f"PRAGMA table_info({table})"))
                pk_columns = [r[1] for r in rows if r[5] >= 1]
                assert pk_columns == [pk_col], f"{table}: expected PK [{pk_col}], got {pk_columns}"
            for non_pk_table in ("partsupp", "lineitem"):
                rows = list(conn.execute(f"PRAGMA table_info({non_pk_table})"))
                pk_columns = [r[1] for r in rows if r[5] >= 1]
                assert pk_columns == [], f"{non_pk_table}: expected no PK, got {pk_columns}"
        finally:
            conn.close()

    def test_read_dataset_properties_char_type_tinyint(self) -> None:
        ddl = (
            "CREATE TABLE lineitem (l_returnflag TINYINT NOT NULL, l_linestatus TINYINT NOT NULL);"
            " CREATE TABLE orders (o_orderstatus TINYINT NOT NULL);"
        )
        props = get_suite(TpchSuite.NAME).read_dataset_properties(ddl)
        assert props["char_type"] == "char"

    def test_read_dataset_properties_char_type_varchar(self) -> None:
        ddl = (
            "CREATE TABLE lineitem (l_returnflag VARCHAR NOT NULL, l_linestatus VARCHAR NOT NULL);"
            " CREATE TABLE orders (o_orderstatus VARCHAR NOT NULL);"
        )
        props = get_suite(TpchSuite.NAME).read_dataset_properties(ddl)
        assert props["char_type"] == "text"

    def test_read_dataset_properties_char_type_mixed_warns(self, caplog) -> None:
        ddl = (
            "CREATE TABLE lineitem (l_returnflag TINYINT NOT NULL, l_linestatus VARCHAR NOT NULL);"
            " CREATE TABLE orders (o_orderstatus TINYINT NOT NULL);"
        )
        with caplog.at_level("WARNING"):
            props = get_suite(TpchSuite.NAME).read_dataset_properties(ddl)
        assert props["char_type"] == "text"
        assert any("Mixed" in rec.message for rec in caplog.records)

    def test_read_dataset_properties_identifier_type_int32(self) -> None:
        ddl = (
            "CREATE TABLE lineitem (l_orderkey INTEGER NOT NULL, l_returnflag TINYINT NOT NULL,"
            " l_linestatus TINYINT NOT NULL);"
            " CREATE TABLE orders (o_orderstatus TINYINT NOT NULL);"
        )
        props = get_suite(TpchSuite.NAME).read_dataset_properties(ddl)
        assert props["identifier_type"] == "int32"

    def test_read_dataset_properties_identifier_type_int64(self) -> None:
        ddl = (
            "CREATE TABLE lineitem (l_orderkey BIGINT NOT NULL, l_returnflag TINYINT NOT NULL,"
            " l_linestatus TINYINT NOT NULL);"
            " CREATE TABLE orders (o_orderstatus TINYINT NOT NULL);"
        )
        props = get_suite(TpchSuite.NAME).read_dataset_properties(ddl)
        assert props["identifier_type"] == "int64"

    def test_query_sql_tpch(self) -> None:
        suite = get_suite(TpchSuite.NAME)
        for i in range(1, 23):
            sql = suite.query_sql(str(i))
            assert "SELECT" in sql.upper(), f"Q{i} missing SELECT"

    def test_solution_file_variant_mapping(self) -> None:
        suite = get_suite(TpchSuite.NAME)
        assert suite.solution_file("Q3_fused_filter", Path("/sol")) == Path("/sol/q3.parquet")
        assert suite.solution_file("10_fused_filter", Path("/sol")) == Path("/sol/q10.parquet")
        assert suite.solution_file("Q1", Path("/sol")) == Path("/sol/q1.parquet")

    def test_solution_file_none_without_dir(self) -> None:
        assert get_suite(TpchSuite.NAME).solution_file("Q1", None) is None

    def test_query_file_with_sql_dir(self) -> None:
        assert get_suite(TpchSuite.NAME).query_file("3", Path("/sql")) == Path("/sql/q3.sql")

    def test_query_file_without_sql_dir_returns_none(self) -> None:
        assert get_suite(TpchSuite.NAME).query_file("3", None) is None

    def test_base_query_name_maps_variants(self) -> None:
        suite = get_suite(TpchSuite.NAME)
        assert suite.base_query_name("2_fused_filter") == "2"
        assert suite.base_query_name("q10_fused_filter") == "10"
        assert suite.base_query_name("3") == "3"


class TestBaseSuite:
    """Tests for the base Suite fallback."""

    def test_unsupported_source_raises(self) -> None:
        with pytest.raises(ValueError):
            get_suite("TPC-DS").available_queries(QuerySource.HANDCODED, None)

    def test_query_sql_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            get_suite("TPC-DS").query_sql("1")

    def test_tables_lists_subdirs(self, tmp_path: Path) -> None:
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()
        (tmp_path / "file.txt").touch()
        names = [t[0] for t in get_suite("TPC-DS").tables(tmp_path)]
        assert "alpha" in names
        assert "beta" in names
        assert "file.txt" not in names

    def test_schema_no_builtin_raises(self) -> None:
        with pytest.raises(NoBuiltInDDL):
            get_suite("TPC-DS").schema(None)

    def test_schema_override_missing_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_suite("TPC-DS").schema(Path("/nonexistent.sql"))

    def test_base_query_name_is_identity(self) -> None:
        assert get_suite("TPC-DS").base_query_name("2_fused_filter") == "2_fused_filter"


class TestNarrowDdl:
    """Tests for narrow_ddl column projection of a CREATE TABLE script."""

    _DDL = (
        "CREATE TABLE t (\n"
        "  a BIGINT NOT NULL,\n"
        "  b VARCHAR NOT NULL,\n"
        "  c INTEGER NOT NULL,\n"
        "  PRIMARY KEY (a)\n"
        ");\n"
    )

    def test_keeps_only_required_columns_in_order_with_types(self) -> None:
        from gqe_bench.suites.base import narrow_ddl

        out = narrow_ddl(self._DDL, {"t": {"a", "c"}})
        assert "a BIGINT NOT NULL" in out
        assert "c INTEGER NOT NULL" in out
        assert "b VARCHAR" not in out
        # Surviving single-column PK is preserved.
        assert "PRIMARY KEY (a)" in out
        # Column order follows the DDL (a before c).
        assert out.index("a BIGINT") < out.index("c INTEGER")

    def test_drops_pk_when_key_column_removed(self) -> None:
        from gqe_bench.suites.base import narrow_ddl

        out = narrow_ddl(self._DDL, {"t": {"b", "c"}})
        assert "PRIMARY KEY" not in out
        assert "a BIGINT" not in out
        assert "b VARCHAR NOT NULL" in out

    def test_table_absent_from_map_keeps_all_columns(self) -> None:
        from gqe_bench.suites.base import narrow_ddl

        out = narrow_ddl(self._DDL, {})
        assert "a BIGINT NOT NULL" in out
        assert "b VARCHAR NOT NULL" in out
        assert "c INTEGER NOT NULL" in out


class TestDecimalTypeSplit:
    """The DDL declares DECIMAL for every run and the engine resolves it; only the
    plan-side types follow the knob. These two assertions are that split."""

    def _schemas(self) -> tuple[object, object]:
        from gqe_bench.suites.tpch.table_schema import TpchTableSchema

        return (
            TpchTableSchema("int64", "double"),
            TpchTableSchema("int64", "decimal"),
        )

    def test_ddl_is_identical_across_representations(self) -> None:
        double_schema, decimal_schema = self._schemas()
        assert double_schema.to_ddl() == decimal_schema.to_ddl()

    def test_ddl_declares_decimal_not_double(self) -> None:
        double_schema, _ = self._schemas()
        ddl = double_schema.to_ddl()
        assert "l_quantity DECIMAL(15,2) NOT NULL" in ddl
        assert "DOUBLE PRECISION" not in ddl

    def test_column_types_follow_the_representation(self) -> None:
        from gqe_bench.physical_plan.expression import DataType, DataTypeId

        double_schema, decimal_schema = self._schemas()
        assert double_schema.column_types("lineitem", ["l_quantity"]) == [
            DataType(DataTypeId.FLOAT64)
        ]
        assert decimal_schema.column_types("lineitem", ["l_quantity"]) == [
            DataType(DataTypeId.DECIMAL64, -2)
        ]

    def test_non_decimal_columns_are_unaffected(self) -> None:
        from gqe_bench.physical_plan.expression import DataType, DataTypeId

        double_schema, decimal_schema = self._schemas()
        for schema in (double_schema, decimal_schema):
            assert schema.column_types("lineitem", ["l_orderkey"]) == [DataType(DataTypeId.INT64)]

    def test_unknown_representation_raises(self) -> None:
        from gqe_bench.suites.tpch.table_schema import TpchTableSchema

        with pytest.raises(ValueError, match="Unknown decimal_type"):
            TpchTableSchema("int64", "fixed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
