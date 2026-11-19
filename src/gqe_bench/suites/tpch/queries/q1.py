# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gqe_bench.physical_plan.expression import (
    Cast,
    DateLiteral,
    DecimalLiteral,
)
from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.relation import Relation
from gqe_bench.schema import (
    LEGACY_TASKGRAPH_EXECUTION_BACKEND,
    QEP_TASKGRAPH_EXECUTION_BACKEND,
)
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def build_plan(
    schema: TpchTableSchema,
    scale_factor: float = 1.0,
    execution_backend: str = LEGACY_TASKGRAPH_EXECUTION_BACKEND,
) -> Relation:
    """Build the physical plan for TPC-H Q1 (pricing summary report)."""
    lineitem = schema.read(
        "lineitem",
        [
            "l_shipdate",
            "l_discount",
            "l_quantity",
            "l_extendedprice",
            "l_returnflag",
            "l_linestatus",
            "l_tax",
        ],
        CR(10) <= DateLiteral("1998-09-02"),
    )

    one = DecimalLiteral("1", schema.decimal_column_type)

    if schema.is_fixed_point:
        decimal128_type = schema.decimal128_type()
        lineitem = lineitem.project(
            [
                CR(0),
                Cast(CR(1), decimal128_type),
                Cast(CR(2), decimal128_type),
                Cast(CR(3), decimal128_type),
                CR(4),
                CR(5),
                Cast(CR(6), decimal128_type),
            ]
        )

    tax_factor = one + CR(6)
    discounted_price = CR(3) * (one - CR(1))
    charge = discounted_price * tax_factor
    shipdate_condition = CR(0) <= DateLiteral("1998-09-02")
    returnflag, linestatus = CR(4), CR(5)
    discount, quantity, extendedprice = CR(1), CR(2), CR(3)
    is_qep = execution_backend == QEP_TASKGRAPH_EXECUTION_BACKEND
    if is_qep:
        # Materialize the discounted price once and derive the charge from it, rather than
        # evaluating extendedprice * (1 - discount) a second time inside the charge expression.
        lineitem = lineitem.project(
            [CR(0), CR(1), CR(2), CR(3), CR(4), CR(5), CR(6), discounted_price]
        )
        lineitem = lineitem.project(
            [CR(0), CR(1), CR(2), CR(3), CR(4), CR(5), CR(6), CR(7), CR(7) * tax_factor]
        )
        # Filter on l_shipdate and keep only the columns the aggregate reads, dropping
        # l_shipdate and l_tax. The filter follows the projections because a QEP projection
        # over masked input nulls the rejected rows, and the perfect-hash aggregate rejects
        # nullable input.
        lineitem = lineitem.filter(shipdate_condition, [1, 2, 3, 4, 5, 7, 8])
        discount, quantity, extendedprice = CR(0), CR(1), CR(2)
        returnflag, linestatus = CR(3), CR(4)
        discounted_price = CR(5)
        charge = CR(6)

    agg = lineitem.aggregate(
        [returnflag, linestatus],
        [
            ("sum", quantity),
            ("sum", extendedprice),
            ("sum", discounted_price),
            ("sum", charge),
            ("sum", discount),
            ("count_all", discount),
        ],
        None if is_qep else shipdate_condition,
        perfect_hashing=True,
    )

    project = agg.project(
        [
            CR(0),
            CR(1),
            CR(2),
            CR(3),
            CR(4),
            CR(5),
            CR(2) / CR(7),
            CR(3) / CR(7),
            CR(6) / CR(7),
            CR(7),
        ]
    )

    return project.sort(
        [
            (CR(0), "ascending", "before"),
            (CR(1), "ascending", "before"),
        ]
    )
