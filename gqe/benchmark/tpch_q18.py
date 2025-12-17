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
from gqe.expression import Literal
from gqe.lib import UniqueKeysPolicy
from gqe.table_definition import TPCHTableDefinitions

"""
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


class tpch_q18(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # After this operation, `lineitem` contains [l_orderkey, sum(l_quantity)]
        lineitem = (
            read("lineitem", ["l_orderkey", "l_quantity"], None, table_defs)
            .aggregate([CR(0)], [("sum", CR(1))], perfect_hashing=True)
            .filter(CR(1) > Literal(300.0), [0, 1])
        )

        customer = read("customer", ["c_custkey", "c_name"], None, table_defs)

        # After this operation, `orders` contains
        # [o_orderkey, o_custkey, o_orderdate, o_totalprice, sum(l_quantity)]
        orders = read(
            "orders",
            ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"],
            None,
            table_defs,
        ).broadcast_join(
            lineitem,
            CR(0) == CR(4),
            [0, 1, 2, 3, 5],
            unique_keys_policy=UniqueKeysPolicy.left,
            perfect_hashing=True,
        )

        # After this operation, `orders` contains
        # [o_orderkey, c_custkey, o_orderdate, o_totalprice, sum(l_quantity), c_name]
        orders = orders.broadcast_join(
            customer,
            CR(1) == CR(5),
            [0, 1, 2, 3, 4, 6],
            unique_keys_policy=UniqueKeysPolicy.right,
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
