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
from gqe.lib import UniqueKeysPolicy
from gqe.table_definition import TPCHTableDefinitions

"""
with revenue (supplier_no, total_revenue) as (
        select
                l_suppkey,
                sum(l_extendedprice * (1 - l_discount))
        from
                lineitem
        where
                l_shipdate >= date '1996-01-01'
                and l_shipdate < date '1996-01-01' + interval '3' month
        group by
                l_suppkey)
select
        s_suppkey,
        s_name,
        s_address,
        s_phone,
        total_revenue
from
        supplier,
        revenue
where
        s_suppkey = supplier_no
        and total_revenue = (
                select
                        max(total_revenue)
                from
                        revenue
        )
order by
        s_suppkey

"""


class tpch_q15(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        lineitem = read(
            "lineitem",
            ["l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"],
            (CR(10) >= DateLiteral("1996-01-01")) & (CR(10) <= DateLiteral("1996-03-31")),
            table_defs,
        ).filter(
            (CR(1) >= DateLiteral("1996-01-01")) & (CR(1) <= DateLiteral("1996-03-31")),
            [0, 2, 3],
        )

        revenue = lineitem.aggregate(
            [CR(0)], [("sum", CR(1) * (Literal(1.0) - CR(2)))], perfect_hashing=True
        )

        max_revenue = revenue.aggregate([], [("max", CR(1))], perfect_hashing=True)

        l_max_revenue = revenue.broadcast_join(
            max_revenue,
            (CR(1) == CR(2)),
            [0, 1],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        supplier = read(
            "supplier",
            ["s_suppkey", "s_name", "s_address", "s_phone"],
            None,
            table_defs,
        )

        unsorted_output = supplier.broadcast_join(
            l_max_revenue,
            (CR(0) == CR(4)),
            [0, 1, 2, 3, 5],
            unique_keys_policy=UniqueKeysPolicy.left,
            perfect_hashing=True,
        )

        return unsorted_output.sort([(CR(0), "ascending", "before")])
