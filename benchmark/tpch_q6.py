# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import Catalog, read, execute
from gqe.expression import ColumnReference as CR
from gqe.expression import Literal, DateLiteral
import argparse


'''
select
        sum(l_extendedprice * l_discount) as revenue
from
        lineitem
where
        l_shipdate >= date '1994-01-01'
        and l_shipdate < date '1994-01-01' + interval '1' year
        and l_discount between 0.06 - 0.01 and 0.06 + 0.01
        and l_quantity < 24
'''


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("location", help="TPC-H dataset location")
    args = arg_parser.parse_args()

    catalog = Catalog()
    catalog.register_tpch(args.location, "memory")

    lineitem = read("lineitem", ["l_shipdate", "l_discount", "l_quantity", "l_extendedprice"])

    # l_shipdate >= date '1994-01-01'
    # and l_shipdate < date '1994-01-01' + interval '1' year
    # and l_discount between 0.06 - 0.01 and 0.06 + 0.01
    # and l_quantity < 24
    lineitem = lineitem.filter(
        (CR(0) >= DateLiteral("1994-01-01")) &
        (CR(0) < DateLiteral("1995-01-01")) &
        (CR(1) >= Literal(0.05)) &
        (CR(1) <= Literal(0.07)) &
        (CR(2) < Literal(24.0)))

    # sum(l_extendedprice * l_discount) as revenue
    lineitem_sum = lineitem.aggregate([], [("sum", CR(3) * CR(1))])

    execute(catalog, lineitem_sum, output_result=True)


if __name__ == "__main__":
    main()
