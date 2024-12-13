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


"""
SQL string: 

select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
from
        lineitem
where
        l_shipdate <= date '1998-12-01' - interval '90' day
group by
        l_returnflag,
        l_linestatus
order by
        l_returnflag,
        l_linestatus

List of optimization over substrait plans:
- When we filter lineitem on l_shipdate, we don't materialize l_shipdate in the output using projection indices.

- For aggregations we reuse the sum and count for values. 
  
  For ex. we need to calculate sum(l_quantity) and avg(l_quantity).
  We could instead calculate sum(l_quantity), count(l_quantity) and then calculate avg through these two columns. 
  GQE effectively already does this via it's multi-partition aggregation implementation.
  
  Additionally, the COUNT(*) == COUNT(l_quantity) as TPC-H columns are guaranteed to be NOT NULL

  We can then reuse the COUNT(*) values across averages of different columns 
  and sum(l_quantity) across sum(l_quantity) and avg(l_quantity)
"""


class tpch_q1(Query):
    def root_relation(self):
        lineitem = read(
            "lineitem",
            [
                "l_shipdate",
                "l_discount",
                "l_quantity",
                "l_extendedprice",
                "l_returnflag",
                "l_linestatus",
                "l_tax",
            ],
        )

        # [ "l_discount", "l_quantity", "l_extendedprice", "l_returnflag", "l_linestatus", "l_tax"]
        lineitem = lineitem.filter(
            (CR(0) <= DateLiteral("1998-09-02")), [1, 2, 3, 4, 5, 6]
        )

        agg = lineitem.aggregate(
            [CR(3), CR(4)],
            [
                ("sum", CR(1)),
                ("sum", CR(2)),
                ("sum", CR(2) * (Literal(1.0) - CR(0))),
                ("sum", CR(2) * (Literal(1.0) - CR(0)) * (Literal(1.0) + CR(5))),
                ("sum", CR(0)),
                ("count_all", CR(0)),
            ],
        )

        project = agg.project(
            [
                (CR(0)),
                (CR(1)),
                (CR(2)),
                (CR(3)),
                CR(4),
                CR(5),
                (CR(2) / CR(7)),
                (CR(3) / CR(7)),
                (CR(6) / CR(7)),
                (CR(7)),
            ]
        )

        sorted_output = project.sort(
            [(CR(0), "ascending", "before"), (CR(1), "ascending", "before")]
        )

        return sorted_output
