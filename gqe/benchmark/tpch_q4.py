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
from gqe.expression import DateLiteral
from gqe.benchmark.query import Query
from gqe.table_definition import TPCHTableDefinitions

"""
select
        o_orderpriority,
        count(*) as order_count
from
        orders
where
        o_orderdate >= date '1993-07-01'
        and o_orderdate < date '1993-07-01' + interval '3' month
        and exists (
                select
                        *
                from
                        lineitem
                where
                        l_orderkey = o_orderkey
                        and l_commitdate < l_receiptdate
        )
group by
        o_orderpriority
order by
        o_orderpriority
"""


class tpch_q4(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        orders = read(
            "orders",
            ["o_orderkey", "o_orderdate", "o_orderpriority"],
            (CR(4) >= DateLiteral("1993-07-01")) & (CR(4) < DateLiteral("1993-10-01")),
            table_defs,
        )

        # o_orderdate >= date '1993-07-01' and o_orderdate < date '1993-07-01' + interval '3' month
        # After this operation, `orders` has column ["o_orderkey", "o_orderpriority"]
        orders = orders.filter(
            (CR(1) >= DateLiteral("1993-07-01")) & (CR(1) < DateLiteral("1993-10-01")),
            [0, 2],
        )

        # l_commitdate < l_receiptdate
        # After this operation, `lineitem` has column ["l_orderkey"]
        lineitem = read(
            "lineitem",
            ["l_orderkey", "l_commitdate", "l_receiptdate"],
            (CR(11) < CR(12)),
            table_defs,
        )
        lineitem = lineitem.filter(CR(1) < CR(2), [0])

        # exists (select * from lineitem where l_orderkey = o_orderkey)
        # Broadcast the left side i.e. `orders` table
        # After this operation, `orders` has column ["o_orderpriority"]
        orders = orders.broadcast_join(lineitem, CR(0) == CR(2), [1], "left_semi", True)

        # group by o_orderpriority
        orders = orders.aggregate(
            [CR(0)], [("count_all", CR(0))], perfect_hashing=False
        )

        # order by o_orderpriority
        return orders.sort([(CR(0), "ascending", "before")])
