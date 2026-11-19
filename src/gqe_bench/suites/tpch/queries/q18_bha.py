# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H Q18 (large volume customer) with a bounded hash aggregate.

Variant of ``q18`` that replaces the lineitem group-by with the engine's
bounded hash aggregate: instead of aggregating each chunk and concatenating
the partials into one merge (whose peak footprint is what makes Q18 fail at
large scale factors), every chunk aggregates directly into a single hash map
pre-allocated from a plan-time bound on the distinct key count.

The bound comes from the TPC-H population spec: ``l_orderkey`` is a foreign
key into ``orders``, whose cardinality is ``1 500 000 * SF`` rows, so the
lineitem group-by produces at most that many distinct keys.

The bounded hash aggregate lowers through the QEP execution backend only, so
this variant requires ``--execution-backend qep_task_graph``.

select
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice,
        sum(l_quantity)
from
        customer,
        orders,
        lineitem
where
        o_orderkey in (
                select
                        l_orderkey
                from
                        lineitem
                group by
                        l_orderkey having
                                sum(l_quantity) > 300
        )
        and c_custkey = o_custkey
        and o_orderkey = l_orderkey
group by
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice
order by
        o_totalprice desc,
        o_orderdate
limit
        100
"""

import math

from gqe_bench.physical_plan.expression import ColumnReference as CR
from gqe_bench.physical_plan.expression import DecimalLiteral
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.schema import QEP_TASKGRAPH_EXECUTION_BACKEND
from gqe_bench.suites.tpch.table_schema import TpchTableSchema

# Rows in `orders` at scale factor 1, per the TPC-H specification (clause 4.2.5).
_ORDERS_ROWS_PER_SCALE_FACTOR = 1_500_000


def orders_cardinality(scale_factor: float) -> int:
    """Return the number of ``orders`` rows at ``scale_factor``, rounded up.

    Fractional scale factors (e.g. ``sf0.01``) are rounded up so the value stays
    an upper bound on the row count.
    """
    return math.ceil(_ORDERS_ROWS_PER_SCALE_FACTOR * scale_factor)


def build_plan(
    schema: TpchTableSchema,
    scale_factor: float = 1.0,
    execution_backend: str = QEP_TASKGRAPH_EXECUTION_BACKEND,
) -> Relation:
    """Build the physical plan for TPC-H Q18 using the bounded hash aggregate."""
    # `l_orderkey` references `o_orderkey`, so the group count is bounded by |orders|.
    group_count_bound = orders_cardinality(scale_factor)

    # A decimal SUM widens to DECIMAL128 in the bounded hash aggregate, so the
    # HAVING threshold has to be widened along with it.
    sum_quantity_type = (
        schema.decimal128_type() if schema.is_fixed_point else schema.decimal_column_type
    )

    # After this operation, `lineitem` contains [l_orderkey, sum(l_quantity)]
    lineitem = (
        schema.read("lineitem", ["l_orderkey", "l_quantity"], None)
        .bounded_hash_aggregate([0], [("sum", 1)], group_count_bound)
        .filter(CR(1) > DecimalLiteral("300", sum_quantity_type), [0, 1])
    )

    customer = schema.read("customer", ["c_custkey", "c_name"], None)

    # After this operation, `orders` contains
    # [o_orderkey, o_custkey, o_orderdate, o_totalprice, sum(l_quantity)]
    #
    # `RIGHT` builds the perfect hash on `lineitem`: the group-by makes `l_orderkey` unique,
    # and the HAVING filter leaves it far smaller than `orders`.
    orders = schema.read(
        "orders",
        ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"],
        None,
    ).broadcast_join(
        lineitem,
        CR(0) == CR(4),
        [0, 1, 2, 3, 5],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # After this operation, `orders` contains
    # [o_orderkey, c_custkey, o_orderdate, o_totalprice, sum(l_quantity), c_name]
    orders = orders.broadcast_join(
        customer,
        CR(1) == CR(5),
        [0, 1, 2, 3, 4, 6],
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
        perfect_hashing=True,
    )

    # After this operation, `orders` contains
    # [c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)]
    orders = (
        orders.aggregate(
            [CR(5), CR(1), CR(0), CR(2), CR(3)],
            [("sum", CR(4))],
            perfect_hashing=False,
        )
        .sort([(CR(4), "descending", "before"), (CR(3), "ascending", "before")])
        .fetch(0, 100)
    )

    return orders
