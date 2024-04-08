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
from gqe.expression import Literal, DateLiteral
from gqe.expression import ColumnReference as CR
from gqe.benchmark.query import Query

'''
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

'''


class tpch_q15(Query):
    def root_relation(self):
        lineitem = read("lineitem", ["l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]) \
            .filter((CR(1) >= DateLiteral("1996-01-01")) & (CR(1) <= DateLiteral("1996-03-31")))

        revenue = lineitem.aggregate([CR(0)], [("sum",  CR(2) * (Literal(1.0) - CR(3)))])

        max_revenue = revenue.aggregate([], [("max", CR(1))])

        l_max_revenue = revenue.broadcast_join(max_revenue, (CR(1) == CR(2)), [0, 1])

        supplier = read("supplier", ["s_suppkey", "s_name", "s_address", "s_phone"])

        unsorted_output = supplier.broadcast_join(l_max_revenue, (CR(0) == CR(4)), [0, 1, 2, 3, 5])

        return unsorted_output.sort([(CR(0), "ascending", "before")])
