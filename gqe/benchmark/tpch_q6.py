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
from gqe.expression import Literal, DateLiteral
from gqe.benchmark.query import Query
from gqe.table_definition import TPCHTableDefinitions


"""
select
        sum(l_extendedprice * l_discount) as revenue
from
        lineitem
where
        l_shipdate >= date '1994-01-01'
        and l_shipdate < date '1994-01-01' + interval '1' year
        and l_discount between 0.06 - 0.01 and 0.06 + 0.01
        and l_quantity < 24
"""


class tpch_q6(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        lineitem = read(
            "lineitem",
            ["l_shipdate", "l_discount", "l_quantity", "l_extendedprice"],
            (CR(10) >= DateLiteral("1994-01-01"))
            & (CR(10) < DateLiteral("1995-01-01"))
            & (CR(6) >= Literal(0.05))
            & (CR(6) <= Literal(0.07))
            & (CR(4) < Literal(24.0)),
            table_defs,
        )

        # l_shipdate >= date '1994-01-01'
        # and l_shipdate < date '1994-01-01' + interval '1' year
        lineitem = lineitem.filter(
            (CR(0) >= DateLiteral("1994-01-01")) & (CR(0) < DateLiteral("1995-01-01")),
            [1, 2, 3],
        )

        # and l_discount between 0.06 - 0.01 and 0.06 + 0.01
        # and l_quantity < 24
        #
        # splitting the filter predicate effectively late materializes this
        # part when using zero-copy
        lineitem = lineitem.filter(
            (CR(0) >= Literal(0.05))
            & (CR(0) <= Literal(0.07))
            & (CR(1) < Literal(24.0)),
            [0, 2],
        )

        # sum(l_extendedprice * l_discount) as revenue
        return lineitem.aggregate([], [("sum", CR(1) * CR(0))], perfect_hashing=True)
