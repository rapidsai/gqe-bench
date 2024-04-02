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
from gqe.expression import Literal
from gqe.expression import ColumnReference as CR
import argparse


'''
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
'''


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("location", help="TPC-H dataset location")
    args = arg_parser.parse_args()

    catalog = Catalog()
    catalog.register_tpch(args.location, "memory")

    part = read("part", ["p_partkey", "p_brand", "p_container"])

    # Filter the part table
    part = part.filter((CR(1) == Literal("Brand#23")) & (CR(2) == Literal("MED BOX")))

    lineitem = read("lineitem", ["l_partkey", "l_quantity", "l_extendedprice"])

    # Join the lineitem with the part table
    # After this operation, `lineitem` has columns ["l_partkey", "l_quantity", "l_extendedprice"]
    lineitem = lineitem.broadcast_join(part, CR(0) == CR(3), [0, 1, 2])

    # Calculate avg(l_quantity) for each `l_partkey`
    # `avg_l_quantity` has columns ["l_partkey", avg(l_quantity)]
    avg_l_quantity = lineitem.aggregate([CR(0)], [("avg", CR(1))])

    # Calculate l_quantity < 0.2 * avg(l_quantity)
    # After this operation, `lineitem` has column ["l_extendedprice"]
    lineitem = lineitem.broadcast_join(
        avg_l_quantity, (CR(0) == CR(3)) & (CR(1) < Literal(0.2) * CR(4)), [2])

    # Calculate sum(l_extendedprice) / 7.0
    sum_l_extendedprice = lineitem.aggregate([], [("sum", CR(0))]).project([CR(0) / Literal(7.0)])

    execute(catalog, sum_l_extendedprice, output_result=True)


if __name__ == "__main__":
    main()
