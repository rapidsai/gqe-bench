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

"""Base class for benchmark suites.

A ``Suite`` is a stateless polymorphic service. It holds no instance
state — every method is a ``@classmethod`` and the registry stores
classes (``type[Suite]``) directly. Subclasses override classmethods to
specialize query enumeration, file mapping, DDL emission, and query
resolution. There are no instances to construct or to share.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gqe_bench.query_source import QuerySource

if TYPE_CHECKING:
    from gqe_bench.schema import Query

logger = logging.getLogger(__name__)


class NoBuiltInDDL(RuntimeError):
    """Raised when a Suite has no built-in DDL and no override was provided."""


class Suite:
    """Stateless base for benchmark suites. Use ``type[Suite]``, never an instance."""

    NAME: str = "_base"

    @classmethod
    def available_queries(
        cls, query_source: QuerySource, query_filter: list[str] | None
    ) -> list[str]:
        raise ValueError("Provide --sql directory for query enumeration")

    @classmethod
    def solution_file(cls, query_name: str, solution_dir: Path | None) -> Path | None:
        if solution_dir is None:
            return None
        return solution_dir / f"{query_name}.parquet"

    @classmethod
    def query_file(cls, query_name: str, sql_dir: Path | None) -> Path | None:
        """Return the path to a user-provided SQL file, or None if no --sql dir."""
        if sql_dir is None:
            return None
        return sql_dir / f"{query_name}.sql"

    @classmethod
    def base_query_name(cls, query_name: str) -> str:
        """Return the base query a name belongs to (identity for the base suite).

        Suites with query variants (e.g. TPC-H's ``2_fused_filter``) override
        this to map a variant back to its base query, so plan grouping treats a
        base query and its variants as one data-load unit.
        """
        return query_name

    @classmethod
    def query_sql(cls, query_name: str, scale_factor: float | None = None) -> str:
        """Generate and return the SQL string for a query on demand.

        ``scale_factor`` is the dataset's scale factor, passed at discovery so
        suites can substitute SF-dependent constants (e.g. TPC-H Q11's
        ``0.0001 / SF`` threshold) before returning the query text.
        """
        raise NotImplementedError(f"{cls.__name__} does not support SQL generation")

    @classmethod
    def tables(cls, dataset: Path) -> list[tuple[str, Path]]:
        return [(d.name, d) for d in sorted(dataset.iterdir()) if d.is_dir()]

    @classmethod
    def to_ddl(cls) -> str:
        """Return the suite's built-in SQL DDL string.

        Subclasses with a built-in TPC-H-style schema override this. The base
        class raises so ``Suite.schema`` can translate the absence into a
        ``NoBuiltInDDL``.
        """
        raise NotImplementedError(f"{cls.__name__} has no built-in DDL")

    @classmethod
    def schema(cls, override: Path | None) -> str:
        """Resolve the DDL string for this suite's load.

        Returns the override file's UTF-8 content if provided; otherwise the
        suite's built-in DDL via ``to_ddl``. Raises ``NoBuiltInDDL`` when no
        override is given and the suite has no built-in DDL.

        The override DDL must be schema-compatible with the suite's table
        layout — column names referenced by handcoded queries must exist
        with types the suite's expression encoding can handle.
        """
        if override is not None:
            logger.info("Using DDL from %s", override)
            return normalize_ddl(override.read_text(encoding="utf-8"))
        try:
            ddl = cls.to_ddl()
        except NotImplementedError as e:
            raise NoBuiltInDDL(f"Suite '{cls.__name__}' has no built-in DDL; pass --schema") from e
        logger.info("Using %s built-in DDL (override with --schema)", cls.__name__)
        return normalize_ddl(ddl)

    @classmethod
    def infer_scale_factor(cls, dataset: Path) -> float:
        """Infer the benchmark scale factor from the dataset path.

        Returns 1.0 (unscaled) when the scale factor cannot be determined.
        """
        return 1.0

    @classmethod
    def read_dataset_properties(cls, ddl: str) -> dict[str, Any]:
        """Read dataset-level properties from a DDL string.

        Returns ``{"identifier_type": ..., "not_null": ...}``. Subclasses that
        need suite-specific properties (e.g., ``char_type``) override this,
        delegate to ``super`` for the shared keys, and add their own.
        """
        any_bigint = False
        for _table, _name, declared, _notnull, _pk in iter_ddl_columns(ddl):
            if "BIGINT" in declared.upper():
                any_bigint = True
                break
        return {
            "identifier_type": "int64" if any_bigint else "int32",
            "not_null": False,
        }

    @classmethod
    def required_columns(
        cls,
        queries: list[Query],
        schema_override: Path | None = None,
    ) -> dict[str, set[str]]:
        """Return per-table column subsets needed by ``queries``.

        For SQL queries: parses ``content`` (UTF-8 SQL) with sqlglot and walks
        the qualified AST. For handcoded queries: walks the serialized
        ``PhysicalRelation`` in ``content`` and collects every ``ReadRelation``.

        ``queries`` must already be resolved (each carries its ``content``);
        resolution happens once at discovery.

        Returns ``{table: {column, ...}}`` with one entry per referenced
        table. Tables in ``cls.tables(...)`` absent from the result can
        be skipped at load time.

        Caller is responsible for deduping ``queries`` if its source has
        repeats (e.g. one Query per QueryParams in a sweep).
        """
        out: dict[str, set[str]] = defaultdict(set)
        ddl = cls.schema(schema_override)
        for query in queries:
            match query.source:
                case QuerySource.SQL:
                    _collect_column_refs_from_sql(ddl, query.content.decode("utf-8"), out)
                case QuerySource.HANDCODED:
                    _collect_column_refs_from_plan(query.content, out)
                case _:
                    raise ValueError(f"Unknown query source: {query.source}")
        return dict(out)

    @classmethod
    def resolve_content(
        cls,
        name: str,
        source: QuerySource,
        sql_file: Path | None,
        scale_factor: float | None,
        identifier_type: str,
        load_all_data: bool,
        decimal_type: str,
        execution_backend: str,
    ) -> bytes:
        """Produce a query's payload bytes at discovery.

        SQL with a user-provided ``sql_file`` → the file's bytes; SQL without
        → generated SQL text (UTF-8); handcoded → the serialized physical
        plan. When ``load_all_data`` is false the load is column-projected, so
        handcoded plans have their read filter indices retargeted to match.

        ``decimal_type`` selects the representation handcoded plans declare for
        the decimal columns and compare against. SQL payloads do not depend on
        it — the catalog DDL declares DECIMAL either way and the engine resolves
        it — so the same text comes back for every value.

        ``execution_backend`` selects the physical-plan shape for handcoded
        queries. SQL payloads do not depend on it.

        Raises ValueError if the query cannot be produced (unknown query name /
        no handcoded builder).
        """
        match source:
            case QuerySource.SQL:
                if sql_file is not None:
                    return sql_file.read_bytes()
                return cls.query_sql(name, scale_factor).encode("utf-8")
            case QuerySource.HANDCODED:
                return cls._serialize_handcoded_plan(
                    name,
                    scale_factor,
                    identifier_type,
                    load_all_data,
                    decimal_type,
                    execution_backend,
                )
            case _:
                raise ValueError(f"Unknown query source: {source}")

    @classmethod
    def _serialize_handcoded_plan(
        cls,
        query_name: str,
        scale_factor: float,
        identifier_type: str,
        load_all_data: bool,
        decimal_type: str,
        execution_backend: str,
    ) -> bytes:
        raise NotImplementedError(f"{cls.__name__} does not support handcoded queries")


def _collect_column_refs_from_sql(ddl: str, sql: str, out: dict[str, set[str]]) -> None:
    """Collect every ``(table, columns)`` pair referenced by ``sql`` under
    ``ddl`` into ``out``.

    Parses with sqlglot, runs ``qualify`` to bind each column reference
    to its FROM-clause source, then walks the AST. Resolves table
    aliases back to real DDL names; columns sourced from a subquery
    alias are skipped — the inner Tables of that subquery are walked on
    their own.

    Note: this is a column-pruning optimization — over-includes are
    safe, under-includes are wrong. ``EXISTS``/``NOT EXISTS`` is the
    only "subquery columns unread" pattern handled; analogous cases
    like ``COUNT(*) FROM (SELECT * FROM t)`` will over-include.
    """
    import sqlglot
    from sqlglot.optimizer.qualify import qualify

    # Rewriting `SELECT *` to `SELECT 1` stops the qualify pass from
    # star-expanding into every column of the inner table.
    parsed = sqlglot.parse_one(sql, dialect="duckdb")
    for exists in parsed.find_all(sqlglot.exp.Exists):
        for star in exists.find_all(sqlglot.exp.Star):
            star.replace(sqlglot.exp.Literal.number(1))

    # Build the {table: {column: type}} dict sqlglot.qualify consumes to
    # disambiguate unqualified column refs. Types unused.
    schema: dict[str, dict[str, str]] = {}
    for table, col, declared, _notnull, _pk in iter_ddl_columns(ddl):
        schema.setdefault(table, {})[col] = declared or "UNKNOWN"

    # qualify sets each Column node's .table to the FROM-clause name it
    # came from — either the real table or a user alias (`l1` for
    # `FROM lineitem l1`).
    qualified = qualify(parsed, schema=schema, dialect="duckdb")

    # Map FROM-clause names back to real DDL tables. Subquery aliases
    # and CTE refs are filtered out (their .name isn't in the schema);
    # columns sourced from them get dropped in the next loop.
    alias_to_table: dict[str, str] = {}
    for t in qualified.find_all(sqlglot.exp.Table):
        if t.name in schema:
            alias_to_table[t.alias_or_name] = t.name

    # Walk every column ref, re-key to its real table, collect.
    for c in qualified.find_all(sqlglot.exp.Column):
        real = alias_to_table.get(c.table)
        if real and c.name:
            out[real].add(c.name)


def _collect_column_refs_from_plan(plan_bytes: bytes, out: dict[str, set[str]]) -> None:
    """Walk a serialized ``PhysicalRelation`` and collect every
    ``ReadRelation``'s ``(table_name, column_names)`` into ``out``.

    Child relations are found from the descriptor rather than a table of field
    names per variant, so a new variant in physical_plan.proto is walked without
    a change here.
    """
    from gqe_bench.physical_plan import proto

    root = proto.physical_plan.PhysicalRelation()
    root.ParseFromString(plan_bytes)

    def visit(rel: Any) -> None:
        kind = rel.WhichOneof("relation")
        if kind is None:
            return
        variant = getattr(rel, kind)
        if kind == "read":
            out[variant.table_name].update(variant.column_names)
        for descriptor in variant.DESCRIPTOR.fields:
            if (
                descriptor.message_type is None
                or descriptor.message_type.name != "PhysicalRelation"
            ):
                continue
            if descriptor.label == descriptor.LABEL_REPEATED:
                for item in getattr(variant, descriptor.name):
                    visit(item)
            elif variant.HasField(descriptor.name):
                visit(getattr(variant, descriptor.name))

    visit(root)


def iter_ddl_columns(ddl: str):
    """Yield ``(table, column, declared_type, notnull_flag, pk_flag)`` for every column in the DDL.

    Single SQLite-backed parser shared by every Suite subclass that
    introspects a DDL string. Callers ignore the fields they don't need.
    """
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(ddl)
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            for row in conn.execute(f"PRAGMA table_info({table})"):
                yield table, row[1], row[2], bool(row[3]), bool(row[5])


def _format_column_line(name: str, decl: str, notnull: bool) -> str:
    """Render one column declaration as it appears inside a CREATE TABLE body."""
    return f"  {name} {decl}{' NOT NULL' if notnull else ''}"


def _primary_key_line(pk_columns: list[str]) -> str:
    """Render a PRIMARY KEY clause line for the given columns."""
    return f"  PRIMARY KEY ({', '.join(pk_columns)})"


def _render_ddl_tables(tables: dict[str, tuple[list[str], list[str], list[str]]]) -> str:
    """Render ``{table: (column_lines, pk_all, pk_kept)}`` into CREATE TABLE statements.

    A ``PRIMARY KEY`` is emitted only when every one of its columns survives
    (``pk_all`` and ``pk_kept`` have equal length).
    """
    parts: list[str] = []
    for table, (lines, pk_all, pk_kept) in tables.items():
        if pk_all and len(pk_kept) == len(pk_all):
            lines.append(_primary_key_line(pk_kept))
        parts.append(f"CREATE TABLE {table} (\n{',\n'.join(lines)}\n);")
    return "\n\n".join(parts) + "\n"


def normalize_ddl(ddl: str) -> str:
    """Re-emit ``ddl`` in canonical shape: one ``CREATE TABLE`` statement per
    table, one column per line, each statement terminated by a lone ``');'``.

    ``narrow_ddl`` and ``append_storage_kind_clause`` both rely on this exact
    shape; call this on any DDL — built-in or from a ``--schema`` override
    file — before it reaches either.

    Only column declarations (name, declared type, ``NOT NULL``) and
    ``PRIMARY KEY`` survive. Other constraints (``CHECK``, foreign keys) are
    dropped — SQLite's ``PRAGMA`` introspection this relies on doesn't expose
    them in a reconstructable form.
    """
    return narrow_ddl(ddl, {})


def narrow_ddl(ddl: str, required: dict[str, set[str]]) -> str:
    """Re-emit ``ddl`` keeping only the columns in ``required`` per table,
    plus every ``PRIMARY KEY`` column regardless of ``required`` — narrowing
    never drops a table's key, so unique-key join optimization stays
    available downstream even when the query doesn't select the key itself.

    Columns are kept in their original DDL order with their declared types
    verbatim; a table absent from ``required`` keeps all of its columns.
    This makes the registered table narrow so downstream loading reads only
    the referenced columns (plus the key) — a column-list COPY for
    in-memory storage, or the external table's own column list for
    ``parquet_file`` storage.

    ``ddl`` is always ``normalize_ddl``'s output (see ``Suite.schema``), so
    every statement is a lone ``CREATE TABLE`` — a column definition per line,
    an optional ``NOT NULL``, and an optional ``PRIMARY KEY``.
    """
    tables: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for table, name, decl, notnull, pk in iter_ddl_columns(ddl):
        lines, pk_all, pk_kept = tables.setdefault(table, ([], [], []))
        keep = required.get(table)
        if pk:
            pk_all.append(name)
        if keep is not None and name not in keep and not pk:
            continue
        lines.append(_format_column_line(name, decl, notnull))
        if pk:
            pk_kept.append(name)
    return _render_ddl_tables(tables)


def append_storage_kind_clause(ddl: str, storage_kind: str) -> str:
    """Append ``WITH (storage_kind = '...')`` to every ``CREATE TABLE`` statement in ``ddl``.

    Only safe to call on ``normalize_ddl`` or ``narrow_ddl`` output, whose
    canonical shape (one statement per table, each terminated by a lone
    ``');'``) is guaranteed by construction — not on arbitrary DDL text.
    """
    escaped = storage_kind.replace("'", "''")
    return ddl.replace(");", f") WITH (storage_kind = '{escaped}');")


def build_external_table_ddl(ddl: str, table_locations: dict[str, Path]) -> str:
    """Re-emit ``ddl`` as ``CREATE EXTERNAL TABLE ... STORED AS PARQUET LOCATION`` statements.

    One statement per table in ``table_locations``, using each table's own
    directory as its ``LOCATION``. Tables absent from ``table_locations`` are
    dropped from the output entirely — parquet-backed tables read directly
    from their location, so a table with nothing to read is skipped. Every
    column of an included table is kept, so its ``PRIMARY KEY`` (if any) is
    always fully present.
    """
    tables: dict[str, tuple[list[str], list[str]]] = {}
    for table, name, decl, notnull, pk in iter_ddl_columns(ddl):
        if table not in table_locations:
            continue
        lines, pk_cols = tables.setdefault(table, ([], []))
        lines.append(_format_column_line(name, decl, notnull))
        if pk:
            pk_cols.append(name)

    parts: list[str] = []
    for table, (lines, pk_cols) in tables.items():
        if pk_cols:
            lines.append(_primary_key_line(pk_cols))
        location = str(table_locations[table]).replace("'", "''")
        parts.append(
            f"CREATE EXTERNAL TABLE {table} (\n{',\n'.join(lines)}\n) "
            f"STORED AS PARQUET LOCATION '{location}';"
        )
    return "\n\n".join(parts) + "\n"
