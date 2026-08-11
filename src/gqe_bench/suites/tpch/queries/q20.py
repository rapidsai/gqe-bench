# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H Q20 (potential part promotion).

select
        s_name,
        s_address
from
        supplier,
        nation
where
        s_suppkey in (
                select
                        ps_suppkey
                from
                        partsupp
                where
                        ps_partkey in (
                                select
                                        p_partkey
                                from
                                        part
                                where
                                        p_name like 'forest%'
                        )
                        and ps_availqty > (
                                select
                                        0.5 * sum(l_quantity)
                                from
                                        lineitem
                                where
                                        l_partkey = ps_partkey
                                        and l_suppkey = ps_suppkey
                                        and l_shipdate >= date '1994-01-01'
                                        and l_shipdate < date '1994-01-01' + interval '1' year
                        )
        )
        and s_nationkey = n_nationkey
        and n_name = 'CANADA'
order by
        s_name
"""

from gqe_bench.physical_plan.expression import (
    Cast,
    DateLiteral,
    DecimalLiteral,
    LikeExpr,
    Literal,
)
from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def _read_filtered_part(schema: TpchTableSchema) -> Relation:
    """p_name like 'forest%' -- selectivity ~1%"""
    part = schema.read("part", ["p_partkey", "p_name"], LikeExpr(CR(1), "forest%"))
    return part.filter(LikeExpr(CR(1), "forest%"), [0])


def _read_filtered_lineitem(schema: TpchTableSchema, part: Relation) -> Relation:
    """Lineitem filtered by date range and semi-joined with part, then aggregated."""
    lineitem = schema.read(
        "lineitem",
        ["l_partkey", "l_suppkey", "l_shipdate", "l_quantity"],
        (CR(10) >= DateLiteral("1994-01-01")) & (CR(10) < DateLiteral("1995-01-01")),
    )
    lineitem = lineitem.filter(
        (CR(2) >= DateLiteral("1994-01-01")) & (CR(2) < DateLiteral("1995-01-01")),
        [0, 1, 3],
    )
    lineitem = lineitem.broadcast_join(part, CR(0) == CR(3), [0, 1, 2], "left_semi")

    # sum(l_quantity) group by (l_partkey, l_suppkey)
    return lineitem.aggregate([CR(0), CR(1)], [("sum", CR(2))], perfect_hashing=True)


def _build_partsupp(schema: TpchTableSchema, part: Relation, lineitem: Relation) -> Relation:
    """Partsupp semi-joined with part and filtered by lineitem aggregates."""
    partsupp = schema.read("partsupp", ["ps_partkey", "ps_suppkey", "ps_availqty"], None)
    partsupp = partsupp.broadcast_join(part, CR(0) == CR(3), [0, 1, 2], "left_semi")

    # `ps_availqty` (int32) is cast to the table's decimal type so the AST `>`
    # compares two decimal operands of the same type id.
    half_literal = DecimalLiteral("0.5", schema.decimal_column_type)
    if schema.is_fixed_point:
        # CAST to decimal type is unsupported by cuDF AST,
        # issue: https://github.com/rapidsai/cudf/issues/22507
        partsupp = partsupp.project([CR(0), CR(1), Cast(CR(2), schema.decimal_column_type)])
        # Pre-materializing because of incorrect result in AST in nested Decimal64 arithmetic
        # Fixed in https://github.com/rapidsai/cudf/pull/22512, needs upgrade to cuDF
        lineitem = lineitem.project([CR(0), CR(1), CR(2) * half_literal])
        threshold_condition = CR(2) > CR(5)
    else:  # float representation
        threshold_condition = Cast(CR(2), schema.decimal_column_type) > CR(5) * half_literal

    return partsupp.broadcast_join(
        lineitem,
        (CR(0) == CR(3)) & (CR(1) == CR(4)) & threshold_condition,
        [1],
        "left_semi",
    )


def _build_supplier(schema: TpchTableSchema) -> Relation:
    """Supplier filtered by nation = 'CANADA'."""
    nation = schema.read("nation", ["n_nationkey", "n_name"], CR(1) == Literal("CANADA"))
    nation = nation.filter(CR(1) == Literal("CANADA"), [0])

    supplier = schema.read(
        "supplier",
        ["s_suppkey", "s_nationkey", "s_name", "s_address"],
        None,
    )
    return supplier.broadcast_join(
        nation,
        CR(1) == CR(4),
        [0, 2, 3],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q20 (potential part promotion)."""
    part = _read_filtered_part(schema)
    lineitem = _read_filtered_lineitem(schema, part)
    partsupp = _build_partsupp(schema, part, lineitem)
    supplier = _build_supplier(schema)

    # s_suppkey in (subquery)
    # After this operation, `supplier` contains columns ["s_name", "s_address"]
    supplier = supplier.broadcast_join(partsupp, CR(0) == CR(3), [1, 2], "left_semi")

    # order by s_name
    return supplier.sort([(CR(0), "ascending", "before")])
