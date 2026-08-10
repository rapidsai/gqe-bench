# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal

from gqe_bench.physical_plan.expression import ColumnReference as CR
from gqe_bench.physical_plan.expression import DecimalLiteral, Literal
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q11 (important stock identification).

    The HAVING threshold is the spec's ``0.0001 / scale_factor``; the
    function applies the scale-factor substitution here.
    """
    nation = schema.read("nation", ["n_nationkey", "n_name"], CR(1) == Literal("GERMANY"))
    nation = nation.filter(CR(1) == Literal("GERMANY"), [0])

    supplier = schema.read("supplier", ["s_suppkey", "s_nationkey"])
    supplier = supplier.broadcast_join(
        nation,
        CR(1) == CR(2),
        [0],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=False,
    )

    partsupp = schema.read(
        "partsupp",
        ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"],
    )
    partsupp = partsupp.broadcast_join(
        supplier,
        CR(1) == CR(4),
        [0, 2, 3],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    fraction_value = Decimal("0.0001") / Decimal(str(scale_factor))
    # Passed the bare type id, not schema.decimal_column_type: the literal then takes its
    # scale from the value rather than the schema's -2, which would round a
    # fraction this small to zero.
    fraction_literal = DecimalLiteral(fraction_value, schema.decimal_column_type.type_id)
    global_sum = partsupp.aggregate([], [("sum", CR(1) * CR(2))], perfect_hashing=True)

    partsupp = partsupp.aggregate([CR(0)], [("sum", CR(1) * CR(2))], perfect_hashing=True)

    # FRACTION = 0.0001 / SF
    # having sum > [FRACTION] * global_sum
    if schema.is_fixed_point:
        # Pre-materializing because of incorrect result in AST in nested Decimal64 arithmetic
        # Fixed in https://github.com/rapidsai/cudf/pull/22512, needs upgrade to cuDF
        global_sum = global_sum.project([CR(0) * fraction_literal])
        partsupp = partsupp.broadcast_join(global_sum, CR(1) > CR(2), [0, 1])
    else:
        partsupp = partsupp.broadcast_join(global_sum, CR(1) > CR(2) * fraction_literal, [0, 1])

    return partsupp.sort([(CR(1), "descending", "before")])
