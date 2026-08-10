# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.expression import (
    LikeExpr,
    Literal,
)
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q16 (parts/supplier relationship)."""
    suppliers_with_complaints = schema.read(
        "supplier",
        ["s_suppkey", "s_comment"],
    ).filter(LikeExpr(CR(1), "%Customer%Complaints%"), [0])

    partsupp = schema.read("partsupp", ["ps_partkey", "ps_suppkey"])

    part = schema.read(
        "part",
        ["p_partkey", "p_brand", "p_type", "p_size"],
    ).filter(
        (
            (CR(3) == Literal(49))
            | (CR(3) == Literal(14))
            | (CR(3) == Literal(23))
            | (CR(3) == Literal(45))
            | (CR(3) == Literal(19))
            | (CR(3) == Literal(3))
            | (CR(3) == Literal(36))
            | (CR(3) == Literal(9))
        )
        & (CR(1) != Literal("Brand#45"))
        & (LikeExpr(CR(2), "MEDIUM POLISHED%") == Literal(False)),
        [0, 1, 2, 3],
    )

    # joined: p_brand(0), p_type(1), p_size(2), ps_suppkey(3)
    joined = part.broadcast_join(
        partsupp,
        CR(0) == CR(4),
        [1, 2, 3, 5],
        unique_keys_policy=UniqueKeysPolicy.LEFT,
        perfect_hashing=True,
    )

    joined = joined.broadcast_join(
        suppliers_with_complaints,
        CR(3) == CR(4),
        [0, 1, 2, 3],
        "left_anti",
    )

    # Two-phase aggregation for count(distinct ps_suppkey)
    distinct_suppliers = joined.aggregate([CR(0), CR(1), CR(2), CR(3)], [])

    return distinct_suppliers.aggregate(
        [CR(0), CR(1), CR(2)],
        [("count_all", CR(0))],
    ).sort(
        [
            (CR(3), "descending", "before"),
            (CR(0), "ascending", "before"),
            (CR(1), "ascending", "before"),
            (CR(2), "ascending", "before"),
        ]
    )
