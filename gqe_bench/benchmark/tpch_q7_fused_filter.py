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
from gqe_bench.expression import ColumnReference as CR
from gqe_bench.expression import DateLiteral, DatePartExpr, Literal
from gqe_bench.lib import UniqueKeysPolicy
from gqe_bench.table_definition import TPCHTableDefinitions

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


class tpch_q7_fused_filter(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # Nation filter predicate rewrite to a conjunctive normal form separable into multiple filters
        # P := (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY') or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
        # <=>
        # P := (n1.n_name = 'FRANCE' or n1.n_name = 'GERMANY') and (n2.n_name = 'GERMANY' or n2.n_name ='FRANCE') and n2.n_name /= n1.n_name
        # TODO: fix column reference indices

        # WHERE n1.n_name = 'FRANCE' or n1.n_name = 'GERMANY'
        nation = read(
            "nation",
            ["n_nationkey", "n_name"],
            (CR(1) == Literal("FRANCE")) | (CR(1) == Literal("GERMANY")),
            table_defs,
        ).filter((CR(1) == Literal("FRANCE")) | (CR(1) == Literal("GERMANY")), [0, 1])

        # join on c_nationkey = n2.n_nationkey
        #   customer has columns ["c_custkey", "c_nationkey"]
        #   returns ["c_custkey", "n_name" as "cust_nation"]
        customer = read("customer", ["c_custkey", "c_nationkey"], None, table_defs).broadcast_join(
            nation,
            CR(1) == CR(2),
            [0, 3],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # join on c_custkey = o_custkey
        #   orders has columns ["o_orderkey", "o_custkey"]
        #   customer has columns ["c_custkey", "cust_nation"]
        #   returns ["o_orderkey", "cust_nation"]
        orders = read("orders", ["o_orderkey", "o_custkey"], None, table_defs).broadcast_join(
            customer,
            CR(1) == CR(2),
            [0, 3],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # WHERE l_shipdate between date '1995-01-01' and date '1996-12-31'
        l1 = read(
            "lineitem",
            ["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"],
            (CR(10) >= DateLiteral("1995-01-01")) & (CR(10) <= DateLiteral("1996-12-31")),
            table_defs,
        )
        # l_shipdate between date '1995-01-01' and date '1996-12-31'
        l1_filter = (CR(2) >= DateLiteral("1995-01-01")) & (CR(2) <= DateLiteral("1996-12-31"))

        # join on o_orderkey = l_orderkey
        #   l1 has columns ["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
        #   orders has columns ["o_orderkey", "cust_nation"]
        #   returns ["cust_nation", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
        l1 = l1.broadcast_join(
            orders,
            CR(0) == CR(5),
            [6, 1, 2, 3, 4],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
            left_filter=l1_filter,
        )

        # join on s_nationkey = n1.n_nationkey
        #   supplier has columns ["s_suppkey", "s_nationkey"]
        #   returns ["s_suppkey", "n_name" as "supp_nation"]
        supplier = read("supplier", ["s_suppkey", "s_nationkey"], None, table_defs).broadcast_join(
            nation,
            CR(1) == CR(2),
            [0, 3],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # join on s_suppkey = l_suppkey and n2.n_name /= n1.n_name
        #   l1 has columns ["cust_nation", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
        #   supplier has columns ["s_suppkey", "supp_nation"]
        #   returns ["supp_nation", "cust_nation", "l_shipdate", "l_extendedprice", "l_discount"]
        #   This is UniqueKeysPolicy.right but not marked in the join
        #   because we cannot yet do unique key joins on
        #   non-equijoins.
        l1 = l1.broadcast_join(supplier, (CR(1) == CR(5)) & (CR(0) != CR(6)), [6, 0, 2, 3, 4])

        # SELECT
        #   supp_nation,
        #   cust_nation,
        #   extract(year from l_shipdate) as l_year,
        #   l_extendedprice * (1 - l_discount) as volume
        l1 = l1.project([CR(0), CR(1), DatePartExpr(CR(2), "year"), CR(3) * (Literal(1.0) - CR(4))])

        # group by
        #   keys ["supp_nation", "cust_nation", "l_year"]
        #   aggs ["sum(volume)"]
        # order by
        #   supp_nation,
        #   cust_nation,
        #   l_year
        l1 = l1.aggregate([CR(0), CR(1), CR(2)], [("sum", CR(3))], perfect_hashing=False).sort(
            [
                (CR(0), "ascending", "before"),
                (CR(1), "ascending", "before"),
                (CR(2), "ascending", "before"),
            ]
        )

        return l1
