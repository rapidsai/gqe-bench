# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H Q17 (small-quantity-order revenue).

select
        sum(l_extendedprice) / 7.0 as avg_yearly
from
        lineitem,
        part
where
        p_partkey = l_partkey
        and p_brand = 'Brand#23'
        and p_container = 'MED BOX'
        and l_quantity < (
                select
                        0.2 * avg(l_quantity)
                from
                        lineitem
                where
                        l_partkey = p_partkey
        )
"""

from gqe_bench.physical_plan.expression import Cast, DecimalLiteral, Literal
from gqe_bench.physical_plan.expression import ColumnReference as CR
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q17 (small-quantity-order revenue)."""
    part = schema.read(
        "part",
        ["p_partkey", "p_brand", "p_container"],
        (CR(3) == Literal("Brand#23")) & (CR(6) == Literal("MED BOX")),
    )

    # Filter the part table
    part = part.filter((CR(1) == Literal("Brand#23")) & (CR(2) == Literal("MED BOX")), [0])

    lineitem = schema.read(
        "lineitem",
        ["l_partkey", "l_quantity", "l_extendedprice"],
        None,
    )

    # Join the lineitem with the part table
    # After this operation, `lineitem` has columns
    # ["l_partkey", "l_quantity", "l_extendedprice"]
    lineitem = lineitem.broadcast_join(
        part,
        CR(0) == CR(3),
        [0, 1, 2],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=False,
    )

    dec_type = schema.decimal_column_type

    avg_l_quantity = lineitem.aggregate([CR(0)], [("avg", CR(1))], perfect_hashing=True)

    # Calculate l_quantity < 0.2 * avg(l_quantity)
    # After this operation, `lineitem` has column ["l_extendedprice"]
    if schema.is_fixed_point:
        # Decimal128 arithmetic is not fully supported in cuDF AST,
        # AVG results in output type of Decimal128
        decimal128_type = schema.decimal128_type()
        avg_l_quantity = avg_l_quantity.project(
            [CR(0), CR(1) * DecimalLiteral("0.2", decimal128_type)]
        )
        lineitem = lineitem.project([CR(0), Cast(CR(1), decimal128_type), CR(2)])
        lineitem = lineitem.broadcast_join(avg_l_quantity, (CR(0) == CR(3)) & (CR(1) < CR(4)), [2])
    else:
        lineitem = lineitem.broadcast_join(
            avg_l_quantity,
            (CR(0) == CR(3)) & (CR(1) < CR(4) * DecimalLiteral("0.2", dec_type)),
            [2],
        )

    # Calculate sum(l_extendedprice) / 7.0
    sum_l_extendedprice = lineitem.aggregate(
        [],
        [("sum", CR(0))],
        perfect_hashing=True,
    ).project([CR(0) / DecimalLiteral("7", dec_type)])

    return sum_l_extendedprice
