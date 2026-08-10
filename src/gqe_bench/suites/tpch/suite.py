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

"""TPC-H suite: query lists, variant→solution mapping, query generation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import duckdb

from gqe_bench.query_source import QuerySource
from gqe_bench.suites.base import Suite, iter_ddl_columns
from gqe_bench.suites.tpch.table_schema import TpchTableSchema

logger = logging.getLogger(__name__)

# Char-typed TPC-H columns. char_type is whichever encoding the DDL declares for
# all three; mixed declarations across them are treated as the non-default and
# logged.
_CHAR_COLUMNS: dict[str, set[str]] = {
    "lineitem": {"l_returnflag", "l_linestatus"},
    "orders": {"o_orderstatus"},
}

_HANDCODED_QUERIES: tuple[str, ...] = (
    "1",
    "2",
    "2_fused_filter",
    "3",
    "3_fused_filter",
    "4",
    "5",
    "6",
    "7",
    "7_fused_filter",
    "9",
    "10",
    "10_fused_filter",
    "11",
    "12",
    "13",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "20_fused_filter",
    "21",
    "22",
)

_ALL_QUERIES: tuple[str, ...] = tuple(str(i) for i in range(1, 23))

_TABLE_NAMES: tuple[str, ...] = (
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
)


def _base_query_name(query_name: str) -> str:
    """``q2_fused_filter`` → ``2``, ``Q10_fused_filter`` → ``10``, ``3`` → ``3``."""
    return query_name.lstrip("qQ").split("_")[0]


# DuckDB emits Q11's SF=1 threshold as `0.0001000000`, inside the subquery as:
#   `sum(ps_supplycost * ps_availqty) * 0.0001000000`.
# Match the multiplication site: `* 0.0001` optionally followed by trailing
# zeros. Tight enough to reject unrelated near-miss constants like 0.00019.
_Q11_THRESHOLD_RE = re.compile(r"\*\s*0\.00010*")


def _apply_scale_factor(base: int, sql: str, scale_factor: float | None) -> str:
    """Substitute scale-factor-dependent constants in DuckDB-generated SQL.

    Q11 is the only standard TPC-H query with a literal SF term; its
    HAVING threshold per the spec is ``0.0001 / SF``. DuckDB's
    ``tpch_queries()`` hardcodes the SF=1 value. For any other SF we
    replace that literal with ``0.0001 / SF``.

    Raises ValueError if ``scale_factor`` is non-positive.
    """
    if scale_factor is None or scale_factor == 1.0:
        return sql
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    if base == 11:
        new_threshold = 0.0001 / scale_factor
        return _Q11_THRESHOLD_RE.sub(f"* {new_threshold:.10f}", sql, count=1)
    return sql


def _classify_char_type(declared: str) -> str:
    """Classify a SQL column type string as ``"char"`` (INT8-compatible), ``"text"`` (VARCHAR/CHAR/TEXT/STRING), or ``"unknown"`` for anything else."""
    upper = declared.strip().upper()
    if upper.startswith("TINYINT") or upper == "INT8":
        return "char"
    if upper.startswith(("VARCHAR", "CHAR", "TEXT")) or upper == "STRING":
        return "text"
    return "unknown"


class TpchSuite(Suite):
    """TPC-H benchmark suite: query list, variant → solution mapping, SQL generation via DuckDB."""

    NAME = "TPC-H"

    @classmethod
    def available_queries(
        cls, query_source: QuerySource, query_filter: list[str] | None
    ) -> list[str]:
        """Return TPC-H query names for ``query_source``, optionally filtered to those in ``query_filter``.

        Raises ValueError for unknown sources.
        """
        match query_source:
            case QuerySource.SQL:
                queries = list(_ALL_QUERIES)
            case QuerySource.HANDCODED:
                queries = list(_HANDCODED_QUERIES)
            case _:
                raise ValueError(f"Unknown query source for TPC-H: {query_source}")

        if query_filter:
            queries = [q for q in queries if q in query_filter]
        return queries

    @classmethod
    def solution_file(cls, query_name: str, solution_dir: Path | None) -> Path | None:
        """Map a query (possibly with variant suffix, e.g. ``2_fused_filter``) to its base-query parquet under ``solution_dir``."""
        if solution_dir is None:
            return None
        return solution_dir / f"q{_base_query_name(query_name)}.parquet"

    @classmethod
    def query_file(cls, query_name: str, sql_dir: Path | None) -> Path | None:
        """Map a query (with optional variant suffix) to its base-query SQL file under ``sql_dir``."""
        return super().query_file(f"q{_base_query_name(query_name)}", sql_dir)

    @classmethod
    def base_query_name(cls, query_name: str) -> str:
        """Map a TPC-H query (possibly a variant like ``2_fused_filter``) to its base query number."""
        return _base_query_name(query_name)

    @classmethod
    def tables(cls, dataset: Path) -> list[tuple[str, Path]]:
        """Return the eight standard TPC-H tables as ``(name, dir)`` pairs under ``dataset``."""
        return [(name, dataset / name) for name in _TABLE_NAMES]

    @classmethod
    def infer_scale_factor(cls, dataset: Path) -> float:
        """Infer TPC-H scale factor from the dataset path.

        Supports integer (``sf100``), fractional (``sf0.01``), and ``k``-suffix
        (``sf1k`` = 1000) spellings. ``Path.resolve()`` follows symlinks when the
        path exists, so the real directory name wins over a misleading symlink
        (e.g. ``/tpch-sf001`` → ``/tpch_scratch/datasets/sf0.01/...`` → 0.01);
        on nonexistent paths it normalizes without dereferencing. Returns 1.0
        (unscaled) when the path carries no scale-factor token.
        """
        match = re.search(r"sf(\d+(?:\.\d+)?)(k?)", str(dataset.resolve()))
        if not match:
            logger.warning("No scale-factor token in dataset path %s; defaulting to 1.0", dataset)
            return 1.0
        value = float(match.group(1))
        if match.group(2) == "k":
            value *= 1000
        return value

    @classmethod
    def query_sql(cls, query_name: str, scale_factor: float | None = None) -> str:
        """Generate TPC-H SQL for a query on demand via DuckDB.

        DuckDB's ``tpch_queries()`` bakes in literals for the default scale
        factor (SF=1). For SF != 1, SF-dependent constants are substituted
        post-hoc so the query matches the TPC-H specification. Q11 is the
        only standard query with a literal SF term (``0.0001 / SF`` threshold).

        Raises ValueError if DuckDB returns no rows for the query number
        (e.g. an unknown query name).
        """
        base = int(_base_query_name(query_name))
        with duckdb.connect() as conn:
            conn.install_extension("tpch")
            conn.load_extension("tpch")
            rows = conn.execute(
                "SELECT query FROM tpch_queries() WHERE query_nr = ?", [base]
            ).fetchall()
        if not rows:
            raise ValueError(f"No TPC-H query found for query number {base}")
        return _apply_scale_factor(base, rows[0][0], scale_factor)

    @classmethod
    def to_ddl(cls) -> str:
        """Built-in TPC-H DDL with id columns declared as ``BIGINT``.

        Matches every shipped TPC-H ``schema.sql`` on hand. To run against
        an id32 dataset, pass ``--schema dataset/schema.sql`` so the
        catalog is populated with the dataset's own (corrected) DDL.
        """
        return TpchTableSchema(identifier_type="int64").to_ddl()

    @classmethod
    def read_dataset_properties(cls, ddl: str) -> dict[str, Any]:
        """Parse DataInfo fields out of the loaded DDL.

        Delegates ``identifier_type`` and ``not_null`` to the base. Adds
        ``char_type``: ``"char"`` if every TPC-H char column declares
        ``TINYINT``, ``"text"`` if every one declares ``VARCHAR`` / ``CHAR``
        / ``TEXT``. Mixed declarations across the three log a warning and
        resolve to ``"text"`` (the non-built-in default).
        """
        props = super().read_dataset_properties(ddl)
        props["char_type"] = cls._detect_char_type(ddl)
        return props

    @classmethod
    def _detect_char_type(cls, ddl: str) -> str:
        observed = [
            _classify_char_type(declared)
            for table, name, declared, _notnull in iter_ddl_columns(ddl)
            if name in _CHAR_COLUMNS.get(table, set())
        ]
        if not observed:
            return "text"
        kinds = set(observed)
        if kinds == {"char"}:
            return "char"
        if kinds == {"text"}:
            return "text"
        logger.warning(
            "Mixed or unrecognized char-type declarations: %s; defaulting to 'text'",
            observed,
        )
        return "text"

    @classmethod
    def _serialize_handcoded_plan(
        cls,
        query_name: str,
        scale_factor: float,
        identifier_type: str,
        load_all_data: bool,
        decimal_type: str,
    ) -> bytes:
        """Serialize a handcoded TPC-H query's physical plan to bytes.

        When ``load_all_data`` is false the load is column-projected, so the
        plan's read partial-filter indices are retargeted from full base-table
        positions to the narrowed load's positions before serialization.

        ``decimal_type`` reaches the plan through the schema: it sets the types
        the reads declare for the decimal columns and the type the builders give
        their decimal literals, so the same query serializes differently per
        representation.
        """
        from gqe_bench.physical_plan.projection import narrow_read_partial_filters
        from gqe_bench.suites.tpch.queries import lookup

        schema = TpchTableSchema(identifier_type=identifier_type, decimal_type=decimal_type)
        root = lookup(query_name)(schema, scale_factor)
        if not load_all_data:
            narrow_read_partial_filters(root, schema.column_orders())
        return root.serialize()
