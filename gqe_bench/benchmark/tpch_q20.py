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

from gqe_bench import read
from gqe_bench.benchmark.query import Query
from gqe_bench.expression import Cast, DateLiteral, LikeExpr, Literal
from gqe_bench.expression import ColumnReference as CR
from gqe_bench.lib import UniqueKeysPolicy
from gqe_bench.table_definition import TPCHTableDefinitions
from gqe_bench.type import Float64

"""
select
        s_name,
        s_address
from
        supplier,
        nation
where
        s_suppkey in (
                select
                        ps_suppkey
                from
                        partsupp
                where
                        ps_partkey in (
                                select
                                        p_partkey
                                from
                                        part
                                where
                                        p_name like 'forest%'
                        )
                        and ps_availqty > (
                                select
                                        0.5 * sum(l_quantity)
                                from
                                        lineitem
                                where
                                        l_partkey = ps_partkey
                                        and l_suppkey = ps_suppkey
                                        and l_shipdate >= date '1994-01-01'
                                        and l_shipdate < date '1994-01-01' + interval '1' year
                        )
        )
        and s_nationkey = n_nationkey
        and n_name = 'CANADA'
order by
        s_name
"""


class tpch_q20(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # p_name like 'forest%'
        # The selectivity of this filter is ~1%
        # After these operations, `part` contains columns ["p_partkey"]
        part = read("part", ["p_partkey", "p_name"], LikeExpr(CR(1), "forest%"), table_defs)
        part = part.filter(LikeExpr(CR(1), "forest%"), [0])

        # l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' year
        # The selectivity of this filter and join is ~0.1%
        # After these operations,
        # `lineitem` contains columns ["l_partkey", "l_suppkey", "l_quantity"]
        lineitem = read(
            "lineitem",
            ["l_partkey", "l_suppkey", "l_shipdate", "l_quantity"],
            (CR(10) >= DateLiteral("1994-01-01")) & (CR(10) < DateLiteral("1995-01-01")),
            table_defs,
        )
        lineitem = lineitem.filter(
            (CR(2) >= DateLiteral("1994-01-01")) & (CR(2) < DateLiteral("1995-01-01")),
            [0, 1, 3],
        )
        lineitem = lineitem.broadcast_join(part, CR(0) == CR(3), [0, 1, 2], "left_semi")

        # sum(l_quantity) group by (l_partkey, l_suppkey)
        # After these operations,
        # `lineitem` contains columns ["l_partkey", "l_suppkey", sum(l_quantity)]
        lineitem = lineitem.aggregate([CR(0), CR(1)], [("sum", CR(2))], perfect_hashing=True)

        # ps_partkey in (subquery) and ps_availqty > (subquery)
        # After these operations, `partsupp` contains columns ["ps_suppkey"]
        partsupp = read("partsupp", ["ps_partkey", "ps_suppkey", "ps_availqty"], None, table_defs)
        partsupp = partsupp.broadcast_join(part, CR(0) == CR(3), [0, 1, 2], "left_semi")
        partsupp = partsupp.broadcast_join(
            lineitem,
            (CR(0) == CR(3)) & (CR(1) == CR(4)) & (Cast(CR(2), Float64()) > Literal(0.5) * CR(5)),
            [1],
            "left_semi",
        )

        # n_name = 'CANADA'
        # After these operations, `nation` contains columns ["n_nationkey"]
        nation = read("nation", ["n_nationkey", "n_name"], CR(1) == Literal("CANADA"), table_defs)
        nation = nation.filter(CR(1) == Literal("CANADA"), [0])

        # s_nationkey = n_nationkey
        # After these operations, `supplier` contains columns ["s_suppkey", "s_name", "s_address"]
        supplier = read(
            "supplier",
            ["s_suppkey", "s_nationkey", "s_name", "s_address"],
            None,
            table_defs,
        )
        supplier = supplier.broadcast_join(
            nation,
            CR(1) == CR(4),
            [0, 2, 3],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # s_suppkey in (subquery)
        # After this operation, `supplier` contains columns ["s_name", "s_address"]
        supplier = supplier.broadcast_join(partsupp, CR(0) == CR(3), [1, 2], "left_semi")

        # order by s_name
        supplier = supplier.sort([(CR(0), "ascending", "before")])

        return supplier
