# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import read
from gqe.expression import ColumnReference as CR
from gqe.expression import Literal, DateLiteral, IfThenElseExpr
from gqe.benchmark.query import Query
from gqe.lib import UniqueKeysPolicy


"""
-- TPC-H Query 12
select
        l_shipmode,
        sum(case
                when o_orderpriority = '1-URGENT'
                        or o_orderpriority = '2-HIGH'
                        then 1
                else 0
        end) as high_line_count,
        sum(case
                when o_orderpriority <> '1-URGENT'
                        and o_orderpriority <> '2-HIGH'
                        then 1
                else 0
        end) as low_line_count
from
        orders,
        lineitem
where
        o_orderkey = l_orderkey
        and l_shipmode in ('MAIL', 'SHIP')
        and l_commitdate < l_receiptdate
        and l_shipdate < l_commitdate
        and l_receiptdate >= date '1994-01-01'
        and l_receiptdate < date '1994-01-01' + interval '1' year
group by
        l_shipmode
order by
        l_shipmode

"""


class tpch_q12(Query):
    def root_relation(self):
        lineitem = read(
            "lineitem",
            ["l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate", "l_orderkey"],
        )

        # After these operations lineitem contains ["l_shipmode", "l_orderkey"]
        # Filter has 1.27% selectivity
        lineitem = lineitem.filter(
            ((CR(0) == Literal("MAIL")) | (CR(0) == Literal("SHIP")))
            & (CR(1) < CR(2))
            & (CR(3) < CR(1))
            & (CR(2) >= DateLiteral("1994-01-01"))
            & (CR(2) <= DateLiteral("1994-12-31")),
            [0, 4],
        )

        orders = read("orders", ["o_orderkey", "o_orderpriority"])

        # After these operations join contains ["l_shipmode", "l_orderkey"]
        # Due to filter, orders table is bigger than lineitem table
        join_out = orders.broadcast_join(lineitem, CR(0) == CR(3), [1, 2], unique_keys_policy=UniqueKeysPolicy.left)

        agg_out = join_out.aggregate(
            [CR(1)],
            [
                (
                    "sum",
                    IfThenElseExpr(
                        (CR(0) == Literal("1-URGENT")) | (CR(0) == Literal("2-HIGH")),
                        Literal(1),
                        Literal(0),
                    ),
                ),
                (
                    "sum",
                    IfThenElseExpr(
                        (CR(0) != Literal("1-URGENT")) & (CR(0) != Literal("2-HIGH")),
                        Literal(1),
                        Literal(0),
                    ),
                ),
            ],
        )

        sort_out = agg_out.sort([(CR(0), "ascending", "before")])

        return sort_out
