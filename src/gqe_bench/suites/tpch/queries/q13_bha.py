# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H Q13 (customer distribution) with a bounded hash aggregate.

Variant of ``q13`` that applies eager right aggregation, following P. Fent and
T. Neumann, "A Practical Approach to Groupjoin and Nested Aggregates", PVLDB
14(11), 2021. The group-by push-down in step 3 is the eager aggregation of
W. P. Yan and P.-A. Larson, "Eager Aggregation and Lazy Aggregation", VLDB 1995,
which Fent and Neumann build on. The logical rewrites from the SQL below are:

1. Push the ON predicate on ``o_comment`` into ``orders``. It references only
   the null-supplying side, so it is a pre-filter on ``orders``.
2. Replace ``count(o_orderkey)`` with ``count(*)``, because ``o_orderkey`` is
   never NULL.
3. Push the ``c_custkey`` group-by below the join onto ``orders`` as a group-by
   on ``o_custkey``, which the join predicate equates with ``c_custkey``.
4. Drop the group-by above the join. ``c_custkey`` is unique, and so is
   ``o_custkey`` after step 3, so the join emits at most one row per group.
5. Compensate for the outer join by mapping a NULL count to zero, because the
   customers without orders now get a NULL count instead of zero. This is
   equivalent to ``coalesce(count, 0)``, or ``CASE WHEN count IS NULL THEN 0
   ELSE count END``, which the plan expresses as ``IF(IS_NULL(count), 0, count)``.

The ``o_custkey`` group-by runs in the bounded hash aggregate, sized to
``|customer| = 150 000 * SF`` because ``o_custkey`` is a foreign key into
``customer``. It lowers through the QEP execution backend only.

select
        c_count,
        count(*) as custdist
from
        (
                select
                        c_custkey,
                        count(o_orderkey)
                from
                        customer left outer join orders on
                                c_custkey = o_custkey
                                and o_comment not like '%special%requests%'
                group by
                        c_custkey
        ) as c_orders (c_custkey, c_count)
group by
        c_count
order by
        custdist desc,
        c_count desc
"""

import math

import numpy as np

from gqe_bench.physical_plan.expression import ColumnReference as CR
from gqe_bench.physical_plan.expression import IfThenElseExpr, IsNullExpr, LikeExpr, Literal
from gqe_bench.physical_plan.relation import Relation, UniqueKeysPolicy
from gqe_bench.schema import QEP_TASKGRAPH_EXECUTION_BACKEND
from gqe_bench.suites.tpch.table_schema import TpchTableSchema

# Rows in `customer` at scale factor 1, per the TPC-H specification (clause 4.2.5).
_CUSTOMER_ROWS_PER_SCALE_FACTOR = 150_000


def customer_cardinality(scale_factor: float) -> int:
    """Return the number of ``customer`` rows at ``scale_factor``, rounded up.

    Fractional scale factors (e.g. ``sf0.01``) are rounded up so the value stays
    an upper bound on the row count.
    """
    return math.ceil(_CUSTOMER_ROWS_PER_SCALE_FACTOR * scale_factor)


def build_plan(
    schema: TpchTableSchema,
    scale_factor: float = 1.0,
    execution_backend: str = QEP_TASKGRAPH_EXECUTION_BACKEND,
) -> Relation:
    """Build the physical plan for TPC-H Q13 using the bounded hash aggregate."""
    # `o_custkey` references `c_custkey`, so the group count is bounded by |customer|.
    group_count_bound = customer_cardinality(scale_factor)

    customer = schema.read("customer", ["c_custkey"])

    # The ON clause's comment predicate references `orders` only, so for a left
    # outer join it equals a pre-filter on `orders`, whose mask the aggregate consumes.
    #
    # After this operation, `orders` contains [o_custkey, count(o_orderkey)].
    #
    # `o_orderkey` is never NULL, so `count(o_orderkey)` is `count_all` over the
    # filtered orders, and the aggregate does not read `o_orderkey` at all.
    orders = (
        schema.read("orders", ["o_custkey", "o_comment"])
        .filter(LikeExpr(CR(1), "%special%requests%") == Literal(False), [0])
        .bounded_hash_aggregate([0], [("count_all", None)], group_count_bound)
    )

    # After this operation, `customer_orders` contains [c_custkey, count(o_orderkey)],
    # with a NULL count for the customers without orders.
    #
    # `RIGHT` marks the aggregate output's keys as unique: the group-by makes
    # `o_custkey` unique, so each customer matches at most one row.
    customer_orders = customer.broadcast_join(
        orders,
        CR(0) == CR(1),
        [0, 2],
        "left",
        unique_keys_policy=UniqueKeysPolicy.RIGHT,
    )

    # The outer join's NULL count is the `count(o_orderkey)` of zero over a
    # NULL-extended row. The literal is INT64 to match the aggregate's count type.
    customer_orders = customer_orders.project(
        [CR(0), IfThenElseExpr(IsNullExpr(CR(1)), Literal(np.int64(0)), CR(1))]
    )

    grouped_c_count = customer_orders.aggregate([CR(1)], [("count_all", CR(1))])

    return grouped_c_count.sort(
        [
            (CR(1), "descending", "before"),
            (CR(0), "descending", "before"),
        ]
    )
