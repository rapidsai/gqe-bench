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
from gqe.expression import DateLiteral, Literal
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
            (CR(0) >= Literal(0.05)) & (CR(0) <= Literal(0.07)) & (CR(1) < Literal(24.0)),
            [0, 2],
        )

        # sum(l_extendedprice * l_discount) as revenue
        return lineitem.aggregate([], [("sum", CR(1) * CR(0))], perfect_hashing=True)
