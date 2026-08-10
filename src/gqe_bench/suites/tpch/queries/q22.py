# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H Q22 (global sales opportunity).

select
        cntrycode,
        count(*) as numcust,
        sum(c_acctbal) as totacctbal
from
        (
                select
                        substring(c_phone from 1 for 2) as cntrycode,
                        c_acctbal
                from
                        customer
                where
                        substring(c_phone from 1 for 2) in
                                ('13', '31', '23', '29', '30', '18', '17')
                        and c_acctbal > (
                                select
                                        avg(c_acctbal)
                                from
                                        customer
                                where
                                        c_acctbal > 0.00
                                        and substring(c_phone from 1 for 2) in
                                                ('13', '31', '23', '29', '30', '18', '17')
                        )
                        and not exists (
                                select
                                        *
                                from
                                        orders
                                where
                                        o_custkey = c_custkey
                        )
        ) as custsale
group by
        cntrycode
order by
        cntrycode
"""

from gqe_bench.physical_plan.expression import (
    Cast,
    DecimalLiteral,
    Literal,
    SubstrExpr,
)
from gqe_bench.physical_plan.expression import (
    ColumnReference as CR,
)
from gqe_bench.physical_plan.relation import Relation
from gqe_bench.suites.tpch.table_schema import TpchTableSchema


def _country_code_filter() -> "Expression":  # noqa: F821
    """Filter condition for the 7 country codes."""
    country_codes = ["13", "31", "23", "29", "30", "18", "17"]
    condition = CR(1) == Literal(country_codes[0])
    for code in country_codes[1:]:
        condition = condition | (CR(1) == Literal(code))
    return condition


def build_plan(schema: TpchTableSchema, scale_factor: float = 1.0) -> Relation:
    """Build the physical plan for TPC-H Q22 (global sales opportunity)."""
    # customer: c_custkey, substring(c_phone, 0, 2), c_acctbal
    customer = schema.read("customer", ["c_custkey", "c_phone", "c_acctbal"], None)
    customer = customer.project([CR(0), SubstrExpr(CR(1), 0, 2), CR(2)])

    # Filter customers with matching country codes
    filter_condition = _country_code_filter()
    filtered_customers = customer.filter(filter_condition, [0, 1, 2])

    # Calculate average account balance for positive balance customers with matching country codes
    pos_balance_customers = filtered_customers.filter(
        CR(2) > DecimalLiteral("0", schema.decimal_column_type), [2]
    )
    avg_acctbal = pos_balance_customers.aggregate([], [("avg", CR(0))], perfect_hashing=False)

    if schema.is_fixed_point:
        decimal128_type = schema.decimal128_type()
        filtered_customers = filtered_customers.project(
            [CR(0), CR(1), Cast(CR(2), decimal128_type)]
        )

    # Filter customers with account balance > average.
    high_balance_customers = filtered_customers.broadcast_join(
        avg_acctbal,
        CR(2) > CR(3),
        [0, 1, 2],
        "left_semi",
    )

    # Find customers with no orders using left anti join
    orders = schema.read("orders", ["o_custkey"], None)
    customers_no_orders = high_balance_customers.broadcast_join(
        orders,
        CR(0) == CR(3),
        [1, 2],
        "left_anti",
        broadcast_left=True,
    )

    # Group by country code and calculate aggregates
    result = customers_no_orders.aggregate(
        [CR(0)],
        [
            ("count_all", CR(0)),
            ("sum", CR(1)),
        ],
        perfect_hashing=False,
    )

    # Order by country code
    return result.sort([(CR(0), "ascending", "before")])
