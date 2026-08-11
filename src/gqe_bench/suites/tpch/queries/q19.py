# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H Q19 (discounted revenue).

select
        sum(l_extendedprice* (1 - l_discount)) as revenue
from
        lineitem,
        part
where
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#12'
                and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                and l_quantity >= 1 and l_quantity <= 1 + 10
                and p_size between 1 and 5
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
        or
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#23'
                and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                and l_quantity >= 10 and l_quantity <= 10 + 10
                and p_size between 1 and 10
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
        or
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#34'
                and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                and l_quantity >= 20 and l_quantity <= 20 + 10
                and p_size between 1 and 15
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
"""

import numpy as np

from gqe_bench.physical_plan.expression import ColumnReference as CR
from gqe_bench.physical_plan.expression import DecimalLiteral, Literal
from gqe_bench.physical_plan.relation import Relation
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def _read_lineitem(schema: TpchTableSchema) -> Relation:
    dec_type = schema.decimal_column_type
    q_1 = DecimalLiteral("1", dec_type)
    q_30 = DecimalLiteral("30", dec_type)

    lineitem = schema.read(
        "lineitem",
        [
            "l_partkey",
            "l_quantity",
            "l_shipmode",
            "l_shipinstruct",
            "l_extendedprice",
            "l_discount",
        ],
        ((CR(14) == Literal("AIR")) | (CR(14) == Literal("AIR REG")))
        & (CR(13) == Literal("DELIVER IN PERSON"))
        & ((CR(4) >= q_1) & (CR(4) <= q_30)),
    )

    # l_quantity between 1 and 30
    # and l_shipmode in ('AIR', 'AIR REG')
    # and l_shipinstruct = 'DELIVER IN PERSON'
    lineitem = lineitem.filter(
        ((CR(2) == Literal("AIR")) | (CR(2) == Literal("AIR REG"))),
        [0, 1, 3, 4, 5],
    )
    lineitem = lineitem.filter(
        (CR(2) == Literal("DELIVER IN PERSON")),
        [0, 1, 3, 4],
    )
    lineitem = lineitem.filter(
        ((CR(1) >= q_1) & (CR(1) <= q_30)),
        [0, 1, 2, 3],
    )
    return lineitem


def _part_filter() -> "Expression":  # noqa: F821
    """Part-side filter for size/brand/container combinations."""
    return (CR(3) >= Literal(1)) & (
        (
            (CR(3) <= Literal(5))
            & (CR(1) == Literal("Brand#12"))
            & (
                (CR(2) == Literal("SM CASE"))
                | (CR(2) == Literal("SM BOX"))
                | (CR(2) == Literal("SM PACK"))
                | (CR(2) == Literal("SM PKG"))
            )
        )
        | (
            (CR(3) <= Literal(10))
            & (CR(1) == Literal("Brand#23"))
            & (
                (CR(2) == Literal("MED BAG"))
                | (CR(2) == Literal("MED BOX"))
                | (CR(2) == Literal("MED PKG"))
                | (CR(2) == Literal("MED PACK"))
            )
        )
        | (
            (CR(3) <= Literal(15))
            & (CR(1) == Literal("Brand#34"))
            & (
                (CR(2) == Literal("LG CASE"))
                | (CR(2) == Literal("LG BOX"))
                | (CR(2) == Literal("LG PACK"))
                | (CR(2) == Literal("LG PKG"))
            )
        )
    )


def _read_part(schema: TpchTableSchema) -> Relation:
    part = schema.read(
        "part",
        ["p_partkey", "p_brand", "p_container", "p_size"],
        (CR(5) >= Literal(1))
        & (
            (
                (CR(5) <= Literal(5))
                & (CR(3) == Literal("Brand#12"))
                & (
                    (CR(6) == Literal("SM CASE"))
                    | (CR(6) == Literal("SM BOX"))
                    | (CR(6) == Literal("SM PACK"))
                    | (CR(6) == Literal("SM PKG"))
                )
            )
            | (
                (CR(5) <= Literal(10))
                & (CR(3) == Literal("Brand#23"))
                & (
                    (CR(6) == Literal("MED BAG"))
                    | (CR(6) == Literal("MED BOX"))
                    | (CR(6) == Literal("MED PKG"))
                    | (CR(6) == Literal("MED PACK"))
                )
            )
            | (
                (CR(5) <= Literal(15))
                & (CR(3) == Literal("Brand#34"))
                & (
                    (CR(6) == Literal("LG CASE"))
                    | (CR(6) == Literal("LG BOX"))
                    | (CR(6) == Literal("LG PACK"))
                    | (CR(6) == Literal("LG PKG"))
                )
            )
        ),
    )
    return part.filter(_part_filter(), [0, 1, 2, 3])


def _join_condition(schema: TpchTableSchema) -> "Expression":  # noqa: F821
    """Join condition combining partkey equality with brand/container/quantity/size checks."""
    dec_type = schema.decimal_column_type
    q_1 = DecimalLiteral("1", dec_type)
    q_10 = DecimalLiteral("10", dec_type)
    q_11 = DecimalLiteral("11", dec_type)
    q_20 = DecimalLiteral("20", dec_type)
    q_30 = DecimalLiteral("30", dec_type)

    return (CR(0) == CR(4)) & (
        (
            (CR(7) <= Literal(np.int32(5)))
            & (CR(5) == Literal("Brand#12"))
            & (
                (CR(6) == Literal("SM CASE"))
                | (CR(6) == Literal("SM BOX"))
                | (CR(6) == Literal("SM PACK"))
                | (CR(6) == Literal("SM PKG"))
            )
            & (CR(1) >= q_1)
            & (CR(1) <= q_11)
        )
        | (
            (CR(7) <= Literal(np.int32(10)))
            & (CR(5) == Literal("Brand#23"))
            & (
                (CR(6) == Literal("MED BAG"))
                | (CR(6) == Literal("MED BOX"))
                | (CR(6) == Literal("MED PKG"))
                | (CR(6) == Literal("MED PACK"))
            )
            & (CR(1) >= q_10)
            & (CR(1) <= q_20)
        )
        | (
            (CR(7) <= Literal(np.int32(15)))
            & (CR(5) == Literal("Brand#34"))
            & (
                (CR(6) == Literal("LG CASE"))
                | (CR(6) == Literal("LG BOX"))
                | (CR(6) == Literal("LG PACK"))
                | (CR(6) == Literal("LG PKG"))
            )
            & (CR(1) >= q_20)
            & (CR(1) <= q_30)
        )
    )


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q19 (discounted revenue)."""
    lineitem = _read_lineitem(schema)
    part = _read_part(schema)

    # lineitem: ['l_partkey', 'l_quantity', 'l_extendedprice', 'l_discount']
    # part: ["p_partkey", "p_brand", "p_container", "p_size"]
    joined = lineitem.broadcast_join(
        part,
        _join_condition(schema),
        [2, 3],  # joined: ['l_extendedprice', 'l_discount']
    )

    # sum(l_extendedprice * (1 - l_discount)) as revenue
    return joined.aggregate(
        [],
        [("sum", CR(0) * (DecimalLiteral("1", schema.decimal_column_type) - CR(1)))],
        perfect_hashing=True,
    )
