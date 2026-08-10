# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H table column definitions for physical plan construction and DDL emission.

This module is the source of truth for the column types the suite uses for
TPC-H. It both:

  - feeds ``read.data_types`` in serialized physical plans for handcoded
    queries (via ``TpchTableSchema.column_types``); and
  - emits a SQL ``CREATE TABLE`` script (via ``TpchTableSchema.to_ddl``) that
    the test harness loads to populate the catalog.

Several columns deviate from a literal reading of the TPC-H spec DDL. Each
deviation is a deliberate workaround for a current gqe / cuDF limitation.
The list is in ``_build_table_schemas``; remove the override (and update
``to_ddl``'s output) when the corresponding limitation is lifted.
"""

from __future__ import annotations

from gqe_bench.physical_plan.expression import DataType, DataTypeId, Expression
from gqe_bench.physical_plan.relation import ReadRelation

_INT32 = DataType(DataTypeId.INT32)
_STRING = DataType(DataTypeId.STRING)
_INT8 = DataType(DataTypeId.INT8)
_DAYS = DataType(DataTypeId.TIMESTAMP_DAYS)

# TPC-H declares DECIMAL(15,2); SQL scale 2 is cuDF scale -2.
_TPC_DECIMAL_SCALE = -2

# The type the decimal columns are *declared* as. ``column_types`` resolves it to
# whichever representation the run selected; ``to_ddl`` always emits DECIMAL(15,2).
_DECIMAL = DataType(DataTypeId.DECIMAL64, _TPC_DECIMAL_SCALE)

# Representation each ``decimal_type`` name selects for the decimal columns.
_DECIMAL_TYPE_BY_NAME: dict[str, DataType] = {
    "double": DataType(DataTypeId.FLOAT64),
    "decimal": _DECIMAL,
}


# DDL keyword for each DataTypeId emitted by to_ddl().
_DDL_KEYWORD: dict[DataTypeId, str] = {
    DataTypeId.INT8: "TINYINT",
    DataTypeId.INT32: "INTEGER",
    DataTypeId.INT64: "BIGINT",
    DataTypeId.FLOAT64: "DOUBLE PRECISION",
    DataTypeId.DECIMAL64: "DECIMAL(15,2)",
    DataTypeId.STRING: "VARCHAR",
    DataTypeId.TIMESTAMP_DAYS: "DATE",
}


# Single-column PRIMARY KEY for each TPC-H table that has one. Composite keys
# (partsupp, lineitem) are intentionally not modelled.
_PRIMARY_KEYS: dict[str, list[str]] = {
    "part": ["p_partkey"],
    "supplier": ["s_suppkey"],
    "customer": ["c_custkey"],
    "orders": ["o_orderkey"],
    "nation": ["n_nationkey"],
    "region": ["r_regionkey"],
}


def _build_table_schemas(id_type: DataType) -> dict[str, dict[str, DataType]]:
    """Build column→type maps for all 8 TPC-H tables.

    Per-column workarounds (deviations from the TPC-H spec DDL):
      - ``l_returnflag`` / ``l_linestatus`` / ``o_orderstatus`` declared
        ``INT8`` instead of ``VARCHAR``. Rationale: fixed-width keys for
        perfect-hash group-by. Remove when gqe supports perfect-hash on
        non-fixed-width keys (gqe#161).
      - ``l_linenumber`` / ``ps_availqty`` declared ``INT32`` instead of
        ``BIGINT``. Rationale: storage shrink; values fit in INT32 by spec.
        Remove if/when storage cost is no longer a concern.
    """
    return {
        "part": {
            "p_partkey": id_type,
            "p_name": _STRING,
            "p_mfgr": _STRING,
            "p_brand": _STRING,
            "p_type": _STRING,
            "p_size": _INT32,
            "p_container": _STRING,
            "p_retailprice": _DECIMAL,
            "p_comment": _STRING,
        },
        "supplier": {
            "s_suppkey": id_type,
            "s_name": _STRING,
            "s_address": _STRING,
            "s_nationkey": id_type,
            "s_phone": _STRING,
            "s_acctbal": _DECIMAL,
            "s_comment": _STRING,
        },
        "partsupp": {
            "ps_partkey": id_type,
            "ps_suppkey": id_type,
            "ps_availqty": _INT32,
            "ps_supplycost": _DECIMAL,
            "ps_comment": _STRING,
        },
        "customer": {
            "c_custkey": id_type,
            "c_name": _STRING,
            "c_address": _STRING,
            "c_nationkey": id_type,
            "c_phone": _STRING,
            "c_acctbal": _DECIMAL,
            "c_mktsegment": _STRING,
            "c_comment": _STRING,
        },
        "orders": {
            "o_orderkey": id_type,
            "o_custkey": id_type,
            "o_orderstatus": _INT8,
            "o_totalprice": _DECIMAL,
            "o_orderdate": _DAYS,
            "o_orderpriority": _STRING,
            "o_clerk": _STRING,
            "o_shippriority": _INT32,
            "o_comment": _STRING,
        },
        "lineitem": {
            "l_orderkey": id_type,
            "l_partkey": id_type,
            "l_suppkey": id_type,
            "l_linenumber": _INT32,
            "l_quantity": _DECIMAL,
            "l_extendedprice": _DECIMAL,
            "l_discount": _DECIMAL,
            "l_tax": _DECIMAL,
            "l_returnflag": _INT8,
            "l_linestatus": _INT8,
            "l_shipdate": _DAYS,
            "l_commitdate": _DAYS,
            "l_receiptdate": _DAYS,
            "l_shipinstruct": _STRING,
            "l_shipmode": _STRING,
            "l_comment": _STRING,
        },
        "nation": {
            "n_nationkey": id_type,
            "n_name": _STRING,
            "n_regionkey": id_type,
            "n_comment": _STRING,
        },
        "region": {
            "r_regionkey": id_type,
            "r_name": _STRING,
            "r_comment": _STRING,
        },
    }


class TpchTableSchema:
    """TPC-H column definitions parameterized by identifier and decimal type."""

    def __init__(self, identifier_type: str = "int64", decimal_type: str = "double") -> None:
        """Build the schema.

        ``identifier_type`` is ``"int64"`` (default) or ``"int32"``.

        ``decimal_type`` is ``"double"`` (default) or ``"decimal"`` and selects the
        representation ``column_types`` reports for the decimal columns. It does not
        reach ``to_ddl``, which declares them ``DECIMAL(15,2)`` either way; the engine
        resolves that declaration against the same knob.

        Raises ValueError if ``decimal_type`` is not one of the known names.
        """
        id_dt = DataType(DataTypeId.INT64 if identifier_type == "int64" else DataTypeId.INT32)
        if decimal_type not in _DECIMAL_TYPE_BY_NAME:
            raise ValueError(
                f"Unknown decimal_type {decimal_type!r}; "
                f"expected one of {sorted(_DECIMAL_TYPE_BY_NAME)}"
            )
        self.decimal_column_type = _DECIMAL_TYPE_BY_NAME[decimal_type]
        self._tables = _build_table_schemas(id_dt)

    def column_types(self, table_name: str, columns: list[str]) -> list[DataType]:
        """Return the plan-side type for each requested column of a table.

        Columns declared decimal resolve to ``self.decimal_column_type``; the rest pass
        through as declared.
        """
        schema = self._tables[table_name]
        return [
            self.decimal_column_type if schema[col] == _DECIMAL else schema[col] for col in columns
        ]

    @property
    def is_fixed_point(self) -> bool:
        """True when the decimal columns carry a cuDF fixed-point type, not FLOAT64.

        ``DataType`` carries a scale only for the fixed-point types, so scale
        presence answers this for any decimal width.
        """
        return self.decimal_column_type.scale is not None

    def decimal128_type(self) -> DataType:
        """Return DECIMAL128 at this schema's decimal scale.

        cuDF's AST yields DECIMAL128 from ``avg`` and from nested decimal
        arithmetic, so a column or literal compared against such a result has to
        be widened to match (rapidsai/cudf#22512).
        """
        return DataType(DataTypeId.DECIMAL128, self.decimal_column_type.scale)

    def column_orders(self) -> dict[str, list[str]]:
        """Return each table's full column list in base-schema (authoring) order."""
        return {table: list(cols) for table, cols in self._tables.items()}

    def read(
        self,
        table_name: str,
        columns: list[str],
        partial_filter: Expression | None = None,
    ) -> ReadRelation:
        """Create a ReadRelation with correct data types for the given table."""
        return ReadRelation(
            table_name,
            columns,
            self.column_types(table_name, columns),
            partial_filter,
        )

    def to_ddl(self) -> str:
        """Emit a SQL ``CREATE TABLE`` script covering all TPC-H tables.

        Column types are the declared types in ``self._tables``, so the decimal
        columns come out ``DECIMAL(15,2)`` for every run. The catalog still agrees
        with the types handcoded plans declare: both follow the run's decimal
        representation, the plans directly through ``column_types`` and the catalog
        through the engine's resolution of this declaration. All columns are
        ``NOT NULL`` (TPC-H spec).
        """
        return (
            "\n\n".join(
                _table_ddl(name, cols, _PRIMARY_KEYS.get(name))
                for name, cols in self._tables.items()
            )
            + "\n"
        )


def _table_ddl(
    table_name: str,
    columns: dict[str, DataType],
    primary_key: list[str] | None = None,
) -> str:
    lines = [f"  {name} {_DDL_KEYWORD[dt.type_id]} NOT NULL" for name, dt in columns.items()]
    if primary_key:
        lines.append(f"  PRIMARY KEY ({', '.join(primary_key)})")
    return f"CREATE TABLE {table_name} (\n{',\n'.join(lines)}\n);"
