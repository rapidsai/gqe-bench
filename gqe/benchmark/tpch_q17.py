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
from gqe.expression import Literal
from gqe.lib import UniqueKeysPolicy
from gqe.table_definition import TPCHTableDefinitions

"""
select
        sum(l_extendedprice) / 7.0 as avg_yearly
from
        lineitem,
        part
where
        p_partkey = l_partkey
        and p_brand = 'Brand#23'
        and p_container = 'MED BOX'
        and l_quantity < (
                select
                        0.2 * avg(l_quantity)
                from
                        lineitem
                where
                        l_partkey = p_partkey
        )
"""


class tpch_q17(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        part = read(
            "part",
            ["p_partkey", "p_brand", "p_container"],
            (CR(3) == Literal("Brand#23")) & (CR(6) == Literal("MED BOX")),
            table_defs,
        )

        # Filter the part table
        part = part.filter((CR(1) == Literal("Brand#23")) & (CR(2) == Literal("MED BOX")), [0])

        lineitem = read(
            "lineitem", ["l_partkey", "l_quantity", "l_extendedprice"], None, table_defs
        )

        # Join the lineitem with the part table
        # After this operation, `lineitem` has columns
        # ["l_partkey", "l_quantity", "l_extendedprice"]
        lineitem = lineitem.broadcast_join(
            part,
            CR(0) == CR(3),
            [0, 1, 2],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=False,
        )

        # Calculate avg(l_quantity) for each `l_partkey`
        # `avg_l_quantity` has columns ["l_partkey", avg(l_quantity)]
        avg_l_quantity = lineitem.aggregate([CR(0)], [("avg", CR(1))], perfect_hashing=True)

        # Calculate l_quantity < 0.2 * avg(l_quantity)
        # After this operation, `lineitem` has column ["l_extendedprice"]
        # This is UniqueKeysPolicy.right but not marked in the join
        # because we cannot yet do unique key joins on non-equijoins.
        lineitem = lineitem.broadcast_join(
            avg_l_quantity, (CR(0) == CR(3)) & (CR(1) < Literal(0.2) * CR(4)), [2]
        )

        # Calculate sum(l_extendedprice) / 7.0
        sum_l_extendedprice = lineitem.aggregate(
            [], [("sum", CR(0))], perfect_hashing=True
        ).project([CR(0) / Literal(7.0)])

        return sum_l_extendedprice
