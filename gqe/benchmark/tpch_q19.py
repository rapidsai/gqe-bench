# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np

from gqe import read
from gqe.benchmark.query import Query
from gqe.expression import ColumnReference as CR
from gqe.expression import Literal
from gqe.table_definition import TPCHTableDefinitions

"""
select
        sum(l_extendedprice* (1 - l_discount)) as revenue
from
        lineitem,
        part
where
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#12'
                and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                and l_quantity >= 1 and l_quantity <= 1 + 10
                and p_size between 1 and 5
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
        or
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#23'
                and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                and l_quantity >= 10 and l_quantity <= 10 + 10
                and p_size between 1 and 10
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
        or
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#34'
                and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                and l_quantity >= 20 and l_quantity <= 20 + 10
                and p_size between 1 and 15
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
"""


class tpch_q19(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        lineitem = read(
            "lineitem",
            [
                "l_partkey",
                "l_quantity",
                "l_shipmode",
                "l_shipinstruct",
                "l_extendedprice",
                "l_discount",
            ],
            ((CR(14) == Literal("AIR")) | (CR(14) == Literal("AIR REG")))
            & (CR(13) == Literal("DELIVER IN PERSON"))
            & ((CR(4) >= Literal(1)) & (CR(4) <= Literal(30))),
            table_defs,
        )

        # l_quantity between 1 and 30
        # and l_shipmode in ('AIR', 'AIR REG')
        # and l_shipinstruct = 'DELIVER IN PERSON'
        # note: late materialization by splitting ANDed conditions
        lineitem = lineitem.filter(
            ((CR(2) == Literal("AIR")) | (CR(2) == Literal("AIR REG"))),
            [
                0,
                1,
                3,
                4,
                5,
            ],  # lineitem: ['l_partkey', 'l_quantity', 'l_shipinstruct', 'l_extendedprice', 'l_discount']
        )
        lineitem = lineitem.filter(
            (CR(2) == Literal("DELIVER IN PERSON")),
            [
                0,
                1,
                3,
                4,
            ],  # lineitem: ['l_partkey', 'l_quantity', 'l_extendedprice', 'l_discount']
        )
        lineitem = lineitem.filter(
            ((CR(1) >= Literal(1)) & (CR(1) <= Literal(30))),
            [
                0,
                1,
                2,
                3,
            ],  # lineitem: ['l_partkey', 'l_quantity', 'l_extendedprice', 'l_discount']
        )

        part = read(
            "part",
            ["p_partkey", "p_brand", "p_container", "p_size"],
            (CR(5) >= Literal(1))
            & (
                (
                    (CR(5) <= Literal(5))
                    & (CR(3) == Literal("Brand#12"))
                    & (
                        (CR(6) == Literal("SM CASE"))
                        | (CR(6) == Literal("SM BOX"))
                        | (CR(6) == Literal("SM PACK"))
                        | (CR(6) == Literal("SM PKG"))
                    )
                )
                | (
                    (CR(5) <= Literal(10))
                    & (CR(3) == Literal("Brand#23"))
                    & (
                        (CR(6) == Literal("MED BAG"))
                        | (CR(6) == Literal("MED BOX"))
                        | (CR(6) == Literal("MED PKG"))
                        | (CR(6) == Literal("MED PACK"))
                    )
                )
                | (
                    (CR(5) <= Literal(15))
                    & (CR(3) == Literal("Brand#34"))
                    & (
                        (CR(6) == Literal("LG CASE"))
                        | (CR(6) == Literal("LG BOX"))
                        | (CR(6) == Literal("LG PACK"))
                        | (CR(6) == Literal("LG PKG"))
                    )
                )
            ),
            table_defs,
        )

        # (p_size between 1 and 5) and (p_brand = 'Brand#12') and (p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'))
        # or (p_size between 1 and 10) and (p_brand = 'Brand#23') and (p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'))
        # or (p_size between 1 and 15) and (p_brand = 'Brand#34') and (p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'))
        # note: condition "p_size >= 1" is factored out
        # note: due to high selectivity of the first ANDed condition, splitting the conditions yield no observable benefit
        part = part.filter(
            (CR(3) >= Literal(1))
            & (
                (
                    (CR(3) <= Literal(5))
                    & (CR(1) == Literal("Brand#12"))
                    & (
                        (CR(2) == Literal("SM CASE"))
                        | (CR(2) == Literal("SM BOX"))
                        | (CR(2) == Literal("SM PACK"))
                        | (CR(2) == Literal("SM PKG"))
                    )
                )
                | (
                    (CR(3) <= Literal(10))
                    & (CR(1) == Literal("Brand#23"))
                    & (
                        (CR(2) == Literal("MED BAG"))
                        | (CR(2) == Literal("MED BOX"))
                        | (CR(2) == Literal("MED PKG"))
                        | (CR(2) == Literal("MED PACK"))
                    )
                )
                | (
                    (CR(3) <= Literal(15))
                    & (CR(1) == Literal("Brand#34"))
                    & (
                        (CR(2) == Literal("LG CASE"))
                        | (CR(2) == Literal("LG BOX"))
                        | (CR(2) == Literal("LG PACK"))
                        | (CR(2) == Literal("LG PKG"))
                    )
                )
            ),
            [0, 1, 2, 3],
        )

        # lineitem: ['l_partkey', 'l_quantity', 'l_extendedprice', 'l_discount']
        # part: ["p_partkey", "p_brand", "p_container", "p_size"]
        joined = lineitem.broadcast_join(
            part,
            (CR(0) == CR(4))
            & (
                (
                    (CR(7) <= Literal(np.int32(5)))
                    & (CR(5) == Literal("Brand#12"))
                    & (
                        (CR(6) == Literal("SM CASE"))
                        | (CR(6) == Literal("SM BOX"))
                        | (CR(6) == Literal("SM PACK"))
                        | (CR(6) == Literal("SM PKG"))
                    )
                    & (CR(1) >= Literal(1.0))
                    & (CR(1) <= Literal(11.0))
                )
                | (
                    (CR(7) <= Literal(np.int32(10)))
                    & (CR(5) == Literal("Brand#23"))
                    & (
                        (CR(6) == Literal("MED BAG"))
                        | (CR(6) == Literal("MED BOX"))
                        | (CR(6) == Literal("MED PKG"))
                        | (CR(6) == Literal("MED PACK"))
                    )
                    & (CR(1) >= Literal(10.0))
                    & (CR(1) <= Literal(20.0))
                )
                | (
                    (CR(7) <= Literal(np.int32(15)))
                    & (CR(5) == Literal("Brand#34"))
                    & (
                        (CR(6) == Literal("LG CASE"))
                        | (CR(6) == Literal("LG BOX"))
                        | (CR(6) == Literal("LG PACK"))
                        | (CR(6) == Literal("LG PKG"))
                    )
                    & (CR(1) >= Literal(20.0))
                    & (CR(1) <= Literal(30.0))
                )
            ),
            [2, 3],  # joined: ['l_extendedprice', 'l_discount']
        )

        # sum(l_extendedprice* (1 - l_discount)) as revenue
        return joined.aggregate([], [("sum", CR(0) * (Literal(1) - CR(1)))], perfect_hashing=True)
