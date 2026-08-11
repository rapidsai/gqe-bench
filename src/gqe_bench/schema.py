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

"""Schema metadata derived from SQL DDL files.

Loads both schema DDL files into an in-memory SQLite database and
uses PRAGMA table_info to discover columns. The DDL is the source
of truth because the most common schema change is adding new
optimization parameters (gqe_parameters) or data-info columns
(gqe_data_info_ext). Most new parameters are binary flags whose DB
column follows the table prefix convention (e.g. p_opt_mark_join,
de_use_cpu_compression), so TableMapping resolves them automatically.
Overrides exist only for legacy columns that predate the naming
convention.
"""

import contextlib
import dataclasses
import importlib.resources
import sqlite3
from pathlib import Path
from typing import Any

import database_benchmarking_tools.experiment as exp
from gqe_bench.query_source import QuerySource


@dataclasses.dataclass(frozen=True)
class TableSchema:
    """Metadata for a single database table."""

    name: str
    prefix: str
    columns: tuple[str, ...]
    defaults: dict[str, Any]  # unprefixed col → parsed DDL DEFAULT (absent if none)
    types: dict[str, str]  # unprefixed col → SQLite type string


def _load_sql_resource(package: str, filename: str) -> str:
    """Load a SQL file from an installed package via importlib.resources."""
    path = importlib.resources.files(package).joinpath(filename)
    return path.read_text(encoding="utf-8")


def _load_base_schema() -> str:
    """Load the base experiment-DB DDL from ``database_benchmarking_tools.sql``."""
    return _load_sql_resource("database_benchmarking_tools.sql", "create_experiment_db.sql")


def _load_gqe_extension() -> str:
    """Load the GQE-specific schema-extension DDL from ``gqe_bench.sql``."""
    return _load_sql_resource("gqe_bench.sql", "system_under_test.sql")


def _parse_dflt_value(raw: str | None) -> Any | None:
    """Parse a PRAGMA table_info dflt_value into a Python literal value."""
    if raw is None:
        return None
    raw = str(raw).strip()
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    if "." in raw:
        return float(raw)
    return int(raw)


def _build_schema() -> tuple[dict[str, TableSchema], dict[str, TableSchema]]:
    """Build the table registry by running both DDL files through an in-memory SQLite and discovering columns + defaults via PRAGMA."""
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(_load_base_schema())
        base_tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        conn.executescript(_load_gqe_extension())
        all_tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        gqe_tables = all_tables - base_tables

        tables: dict[str, TableSchema] = {}
        by_prefix: dict[str, TableSchema] = {}
        for name in all_tables:
            pragma_rows = conn.execute(f"PRAGMA table_info({name})").fetchall()
            if not pragma_rows:
                continue
            cols = [row[1] for row in pragma_rows]
            first_col = cols[0]
            underscore = first_col.find("_")
            prefix = first_col[: underscore + 1] if underscore > 0 else ""
            defaults: dict[str, Any] = {}
            types: dict[str, str] = {}
            for row in pragma_rows:
                col_name = row[1]
                unprefixed = col_name[len(prefix) :]
                types[unprefixed] = row[2]
                if name in gqe_tables:
                    parsed = _parse_dflt_value(row[4])
                    if parsed is not None:
                        defaults[unprefixed] = parsed
            ts = TableSchema(
                name=name,
                prefix=prefix,
                columns=tuple(cols),
                defaults=defaults,
                types=types,
            )
            tables[name] = ts
            if prefix:
                by_prefix[prefix] = ts
        return tables, by_prefix


# Table registry
#
# _TABLES (name → TableSchema) and _TABLES_BY_PREFIX (prefix → TableSchema)
# are nominally lazy-initialized on first access via get_tables(). In
# practice, initialization is eager at import time: the module-level
# constants Q_NAME, SCALE_FACTOR, DATA_INFO_MAPPING, DataInfo, etc. all
# call get_tables() during module load. The guard only prevents redundant
# _build_schema() calls, not deferred init.
#
# _build_schema() opens a transient in-memory SQLite connection (closed
# before returning). The registries live for the process lifetime. Both
# DDL packages (database_benchmarking_tools.sql, gqe_bench.sql) must
# be installed at import time.

_TABLES: dict[str, TableSchema] | None = None
_TABLES_BY_PREFIX: dict[str, TableSchema] | None = None


def get_tables() -> dict[str, TableSchema]:
    """Return the table registry, building it on first call."""
    global _TABLES, _TABLES_BY_PREFIX
    if _TABLES is None:
        _TABLES, _TABLES_BY_PREFIX = _build_schema()
    return _TABLES


def _table_for_prefix(prefix: str) -> TableSchema | None:
    """Look up a table by its column prefix (e.g. 'd_' → data_info)."""
    get_tables()  # ensure initialized
    return _TABLES_BY_PREFIX.get(prefix)


def _column(table_name: str, unprefixed: str) -> str:
    """Build a prefixed column name.

    Example: _column("query_info", "name") → "q_name"

    Raises KeyError if the prefixed column isn't in the table.
    """
    t = get_tables()[table_name]
    col = f"{t.prefix}{unprefixed}"
    if col not in t.columns:
        raise KeyError(f"Column '{col}' not in table '{table_name}'")
    return col


VIEW_BEST_PARAMS_VALIDATED = "gqe_best_parameters_validated"


Q_NAME = _column("query_info", "name")
Q_SOURCE = _column("query_info", "source")
SCALE_FACTOR = _column("data_info", "scale_factor")
AVG_DURATION = "r_avg_duration_s"  # computed alias in _gqe_data_base_view
QUERY_NAME_PREFIX = "Q"  # q_name is stored prefixed; bare everywhere else


class TableMapping:
    """Maps sweep fields to DB columns, auto-derived from PRAGMA.

    Columns are discovered from the table, prefix-stripped to field
    names, with skip_columns filtered out. These are the table-native
    fields used for both sweep iteration and DB insertion.

    Override entries add cross-table fields (from parent tables in
    joined views) and specify their column name directly. These are
    included in sweep_fields and extract(), but excluded from
    to_insert_kwargs() since they belong to a different table.
    """

    def __init__(
        self,
        table_name: str,
        skip_columns: frozenset[str],
        *,
        overrides: dict[str, str],
    ) -> None:
        """Initialize column-name lookups from ``table_name``'s PRAGMA, dropping ``skip_columns`` and adding cross-table ``overrides``."""
        t = get_tables()[table_name]
        auto_fields = tuple(
            col[len(t.prefix) :] for col in t.columns if col[len(t.prefix) :] not in skip_columns
        )
        self._table = t
        self._table_fields = auto_fields
        self._sweep_fields = auto_fields + tuple(overrides.keys())
        self._col_map: dict[str, str] = {f: _column(table_name, f) for f in auto_fields}
        self._col_map.update(overrides)

    @property
    def sweep_fields(self) -> tuple[str, ...]:
        """All sweepable field names (table-native + cross-table overrides)."""
        return self._sweep_fields

    def field_default(self, name: str) -> Any | None:
        """Return the DEFAULT value from the DDL for this sweep field.

        Source of truth: the DEFAULT clause on the column in system_under_test.sql.
        Returns None if the column has no DEFAULT (meaning it must be explicitly
        provided in the JSON5 config or CLI).
        """
        return self._table.defaults.get(name)

    def field_type(self, name: str) -> str:
        """Return the SQLite type string for a sweep field.

        For table-native fields, the type comes from this table's PRAGMA.
        For cross-table override fields, the type is resolved from the
        table that owns the column (matched by column prefix).
        """
        if name in self._table.types:
            return self._table.types[name]
        col_name = self._col_map.get(name, "")
        underscore = col_name.find("_")
        if underscore > 0:
            t = _table_for_prefix(col_name[: underscore + 1])
            if t is not None:
                unprefixed = col_name[len(t.prefix) :]
                return t.types.get(unprefixed, "")
        return ""

    def is_bool_field(self, name: str) -> bool:
        """True if the field is a boolean flag (INTEGER, DEFAULT 0 or 1, name matches bool pattern)."""
        if self._table.types.get(name) != "INTEGER":
            return False
        if self._table.defaults.get(name) not in (0, 1):
            return False
        return name.startswith("use_") or "_use_" in name

    def extract(self, row: dict[str, Any]) -> dict[str, Any]:
        """Extract sweep field values from a DB row into dataclass kwargs."""
        return {f: row[self._col_map[f]] for f in self._sweep_fields}

    def to_insert_kwargs(self, source: object, **extra: Any) -> dict[str, Any]:
        """Build insertion kwargs for this table from a dataclass.

        Only includes table-native fields (not cross-table overrides),
        plus any extra FK fields passed as keyword arguments.
        """
        kwargs = {f: getattr(source, f) for f in self._table_fields}
        kwargs.update(extra)
        return kwargs


def _query_validated_params(db_path: Path) -> list[dict[str, Any]]:
    """Query best validated parameters from a single .db3 file.

    Query names are returned bare -- ``QUERY_NAME_PREFIX`` is removed here so
    callers never see the stored form. Databases written before the prefix
    existed hold bare names and pass through unchanged.
    """
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {VIEW_BEST_PARAMS_VALIDATED}").fetchall()
    params = [dict(r) for r in rows]
    for row in params:
        row[Q_NAME] = row[Q_NAME].removeprefix(QUERY_NAME_PREFIX)
    return params


def query_best_parameters(db_paths: list[Path]) -> list[dict[str, Any]]:
    """Query best validated parameters across one or more .db3 files.

    Each file is queried individually. When multiple files contain the
    same query, the result with the lowest average duration wins.
    """
    best: dict[str, dict[str, Any]] = {}
    for db_path in db_paths:
        for row in _query_validated_params(db_path):
            name = row[Q_NAME]
            if name not in best or row[AVG_DURATION] < best[name][AVG_DURATION]:
                best[name] = row
    return list(best.values())


DATA_INFO_MAPPING = TableMapping(
    "gqe_data_info_ext",
    skip_columns=frozenset({"id", "data_info_id"}),
    overrides={
        "storage_device_kind": "d_storage_device_kind",
        "decimal_type": "d_decimal_type",
    },
)

QUERY_PARAMS_MAPPING = TableMapping(
    "gqe_parameters",
    skip_columns=frozenset({"id", "sut_info_id"}),
    overrides={},
)


_SQLITE_TYPE_MAP: dict[str, type] = {"INTEGER": int, "REAL": float, "TEXT": str}


def _ddl_fields_with_defaults(
    mapping: TableMapping,
) -> list[tuple[str, type, Any]]:
    """Derive dataclass field definitions with DDL defaults from column types."""
    fields: list[tuple[str, type, Any]] = []
    for name in mapping.sweep_fields:
        if mapping.is_bool_field(name):
            py_type = bool
            default = bool(mapping.field_default(name) or 0)
        else:
            sql_type = mapping.field_type(name)
            if sql_type not in _SQLITE_TYPE_MAP:
                raise ValueError(f"Unmapped SQLite type '{sql_type}' for field '{name}'")
            py_type = _SQLITE_TYPE_MAP[sql_type]
            default = mapping.field_default(name)
        if default is not None:
            fields.append((name, py_type, dataclasses.field(default=default)))
        else:
            fields.append((name, py_type))
    return fields


def _generic_str(self: object) -> str:
    parts = [f"{f.name}={getattr(self, f.name)}" for f in dataclasses.fields(self)]
    return ", ".join(parts)


# DataInfo inherits from exp.DataInfo (base schema fields: location,
# scale_factor, identifier_type, etc. all default to None).
# DDL fields get DDL defaults. All fields have defaults, so frozen
# dataclass field ordering is satisfied.
DataInfo = dataclasses.make_dataclass(
    "DataInfo",
    _ddl_fields_with_defaults(DATA_INFO_MAPPING)
    # GQE-invariant experiment metadata (recorded in DB, not runtime config).
    # format: GQE's in-memory columnar representation (vs csv/parquet).
    + [
        ("format", str, dataclasses.field(default="internal")),
    ],
    bases=(exp.DataInfo,),
    frozen=True,
    namespace={"__str__": _generic_str},
)

# QueryParams has no base class in database_benchmarking_tools.
# Sweep-variant fields only (DDL-derived). Identity (name, source) and
# runtime-only fields (reference_file, sql_file) live on Query, not here.
QueryParams = dataclasses.make_dataclass(
    "QueryParams",
    _ddl_fields_with_defaults(QUERY_PARAMS_MAPPING),
    frozen=True,
    namespace={"__str__": _generic_str},
)


@dataclasses.dataclass(frozen=True)
class Query:
    """Per-query identity and resolved payload.

    ``name`` and ``source`` are persisted to ``query_info`` at experiment
    recording time. ``reference_file`` (runtime-only) drives output
    validation. ``content`` is the resolved payload — the SQL text (UTF-8) or
    the serialized physical plan — produced once at discovery; ``source``
    discriminates how to interpret it.
    """

    name: str
    source: QuerySource
    reference_file: Path | None
    content: bytes


@dataclasses.dataclass
class DataLoadGroup:
    """A DataInfo paired with the ``(Query, QueryParams)`` pairs to run.

    Mutable — ``queries`` is consumed (popped) during execution to track
    progress. If the server dies, remaining pairs are resumed after restart.
    """

    data_info: DataInfo
    queries: list[tuple[Query, QueryParams]]
