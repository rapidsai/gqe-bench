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
from gqe.expression import ColumnReference as CR, Literal, DatePartExpr, DateLiteral
from gqe.benchmark.query import Query
from gqe.lib import UniqueKeysPolicy

"""
select
        supp_nation,
        cust_nation,
        l_year,
        sum(volume) as revenue
from
        (select
                        n1.n_name as supp_nation,
                        n2.n_name as cust_nation,
                        extract(year from l_shipdate) as l_year,
                        l_extendedprice * (1 - l_discount) as volume
                from
                        supplier,
                        lineitem,
                        orders,
                        customer,
                        nation n1,
                        nation n2
                where
                        s_suppkey = l_suppkey
                        and o_orderkey = l_orderkey
                        and c_custkey = o_custkey
                        and s_nationkey = n1.n_nationkey
                        and c_nationkey = n2.n_nationkey
                        and (
                                (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
                                or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
                        )
                        and l_shipdate between date '1995-01-01' and date '1996-12-31'
        ) as shipping
group by
        supp_nation,
        cust_nation,
        l_year
order by
        supp_nation,
        cust_nation,
        l_year
"""


class tpch_q7(Query):
    def root_relation(self):
        # Nation filter predicate rewrite to a conjunctive normal form separable into multiple filters
        # P := (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY') or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
        # <=>
        # P := (n1.n_name = 'FRANCE' or n1.n_name = 'GERMANY') and (n2.n_name = 'GERMANY' or n2.n_name ='FRANCE') and n2.n_name /= n1.n_name
        # TODO: fix column reference indices

        # WHERE n1.n_name = 'FRANCE' or n1.n_name = 'GERMANY'
        nation = read("nation", ["n_nationkey", "n_name"]).filter(
            (CR(1) == Literal("FRANCE")) | (CR(1) == Literal("GERMANY")), [0, 1]
        )

        # join on c_nationkey = n2.n_nationkey
        #   customer has columns ["c_custkey", "c_nationkey"]
        #   returns ["c_custkey", "n_name" as "cust_nation"]
        customer = read("customer", ["c_custkey", "c_nationkey"]).broadcast_join(
            nation, CR(1) == CR(2), [0, 3], unique_keys_policy=UniqueKeysPolicy.right
        )

        # join on c_custkey = o_custkey
        #   orders has columns ["o_orderkey", "o_custkey"]
        #   customer has columns ["c_custkey", "cust_nation"]
        #   returns ["o_orderkey", "cust_nation"]
        orders = read("orders", ["o_orderkey", "o_custkey"]).broadcast_join(
            customer, CR(1) == CR(2), [0, 3], unique_keys_policy=UniqueKeysPolicy.right
        )

        # WHERE l_shipdate between date '1995-01-01' and date '1996-12-31'
        l1 = read(
            "lineitem",
            ["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"],
        ).filter(
            (CR(2) >= DateLiteral("1995-01-01")) & (CR(2) <= DateLiteral("1996-12-31")),
            [0, 1, 2, 3, 4],
        )

        # join on o_orderkey = l_orderkey
        #   l1 has columns ["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
        #   orders has columns ["o_orderkey", "cust_nation"]
        #   returns ["cust_nation", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
        l1 = l1.broadcast_join(orders, CR(0) == CR(5), [6, 1, 2, 3, 4], unique_keys_policy=UniqueKeysPolicy.right)

        # join on s_nationkey = n1.n_nationkey
        #   supplier has columns ["s_suppkey", "s_nationkey"]
        #   returns ["s_suppkey", "n_name" as "supp_nation"]
        supplier = read("supplier", ["s_suppkey", "s_nationkey"]).broadcast_join(
            nation, CR(1) == CR(2), [0, 3], unique_keys_policy=UniqueKeysPolicy.right
        )

        # join on s_suppkey = l_suppkey and n2.n_name /= n1.n_name
        #   l1 has columns ["cust_nation", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
        #   supplier has columns ["s_suppkey", "supp_nation"]
        #   returns ["supp_nation", "cust_nation", "l_shipdate", "l_extendedprice", "l_discount"]
        #   This is UniqueKeysPolicy.right but not marked in the join
        #   because we cannot yet do unique key joins on
        #   non-equijoins.
        l1 = l1.broadcast_join(
            supplier, (CR(1) == CR(5)) & (CR(0) != CR(6)), [6, 0, 2, 3, 4]
        )

        # SELECT
        #   supp_nation,
        #   cust_nation,
        #   extract(year from l_shipdate) as l_year,
        #   l_extendedprice * (1 - l_discount) as volume
        l1 = l1.project(
            [CR(0), CR(1), DatePartExpr(CR(2), "year"), CR(3) * (Literal(1.0) - CR(4))]
        )

        # group by
        #   keys ["supp_nation", "cust_nation", "l_year"]
        #   aggs ["sum(volume)"]
        # order by
        #   supp_nation,
        #   cust_nation,
        #   l_year
        l1 = l1.aggregate([CR(0), CR(1), CR(2)], [("sum", CR(3))]).sort(
            [
                (CR(0), "ascending", "before"),
                (CR(1), "ascending", "before"),
                (CR(2), "ascending", "before"),
            ]
        )

        return l1
