# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import read
from gqe.benchmark.query import Query
from gqe.expression import ColumnReference as CR
from gqe.expression import DateLiteral, IfThenElseExpr, Literal
from gqe.lib import UniqueKeysPolicy
from gqe.table_definition import TPCHTableDefinitions

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


class tpch_q12_shuffle(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        lineitem = read(
            "lineitem",
            ["l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate", "l_orderkey"],
            ((CR(14) == Literal("MAIL")) | (CR(14) == Literal("SHIP")))
            & (CR(12) > CR(11))
            & (CR(11) > CR(10))
            & (CR(12) >= DateLiteral("1994-01-01"))
            & (CR(12) <= DateLiteral("1994-12-31")),
            table_defs,
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

        orders = read("orders", ["o_orderkey", "o_orderpriority"], None, table_defs)

        # we need to have the shuffle relation as the child of shuffle joins when necessary,
        # or it will produce wrong results
        lineitem = lineitem.shuffle([CR(1)])  # shuffle cols: l_orderkey
        orders = orders.shuffle([CR(0)])  # shuffle cols: o_orderkey
        # After these operations join contains ["l_shipmode", "l_orderkey"]
        # Due to filter, orders table is bigger than lineitem table
        join_out = orders.shuffle_join(
            lineitem,
            CR(0) == CR(3),
            [1, 2],
            unique_keys_policy=UniqueKeysPolicy.left,
            perfect_hashing=True,
        )

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
            perfect_hashing=False,
        )

        sort_out = agg_out.sort([(CR(0), "ascending", "before")])

        return sort_out
