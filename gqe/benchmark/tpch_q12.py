# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

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


class tpch_q12(Query):
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

        # After these operations join contains ["l_shipmode", "l_orderkey"]
        # Due to filter, orders table is bigger than lineitem table
        join_out = orders.broadcast_join(
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
