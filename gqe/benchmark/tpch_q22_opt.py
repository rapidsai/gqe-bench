# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from gqe.benchmark.query import Query
from gqe.relation import Relation
import gqe.lib

class Q22FusedProjectFilterRelation(Relation):
    """
    Q22 specific fused project and filter relation.
    """

    def __init__(
        self,
        input: Relation
    ):
        self.input = input

    def _to_cpp(self):
        return gqe.lib.q22_fused_project_filter(
            self.input._cpp
        )

class Q22MarkJoinRelation(Relation):
    """
    Q22 specific mark join relation to replace left-anti join.
    """

    def __init__(
        self,
        customer_table: Relation,
        orders_table: Relation
    ):
        self.customer_table = customer_table
        self.orders_table = orders_table

    def _to_cpp(self):
        return gqe.lib.q22_mark_join(
            self.customer_table._cpp,
            self.orders_table._cpp
        )

"""
select
        cntrycode,
        count(*) as numcust,
        sum(c_acctbal) as totacctbal
from
        (
                select
                        substring(c_phone from 1 for 2) as cntrycode,
                        c_acctbal
                from
                        customer
                where
                        substring(c_phone from 1 for 2) in
                                ('13', '31', '23', '29', '30', '18', '17')
                        and c_acctbal > (
                                select
                                        avg(c_acctbal)
                                from
                                        customer
                                where
                                        c_acctbal > 0.00
                                        and substring(c_phone from 1 for 2) in
                                                ('13', '31', '23', '29', '30', '18', '17')
                        )
                        and not exists (
                                select
                                        *
                                from
                                        orders
                                where
                                        o_custkey = c_custkey
                        )
        ) as custsale
group by
        cntrycode
order by
        cntrycode
"""

class tpch_q22_opt(Query):
    def root_relation(self):
        
        # customer: c_custkey, substring(c_phone, 0, 2), c_acctbal
        customer = read("customer", ["c_custkey", "c_phone", "c_acctbal"])
        filtered_customers = Q22FusedProjectFilterRelation(customer)

        # Calculate average account balance for positive balance customers with matching country codes
        avg_acctbal = filtered_customers.aggregate([], [("avg", CR(2))], perfect_hashing=False)
        
        # Filter customers with account balance > average
        high_balance_customers = filtered_customers.broadcast_join(
            avg_acctbal, CR(2) > CR(3), [0, 1, 2], "left_semi"
        )
        
        # Find customers with no orders using left anti join
        orders = read("orders", ["o_custkey"])
        customers_no_orders = Q22MarkJoinRelation(high_balance_customers, orders)

        # Group by country code and calculate aggregates
        # result: cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal
        result = customers_no_orders.aggregate(
            [CR(0)],  # Group by cntrycode
            [
                ("count_all", CR(0)),  # count(*) as numcust
                ("sum", CR(1))         # sum(c_acctbal) as totacctbal
            ],
            perfect_hashing=False
        )
        
        # Order by country code
        result = result.sort([
            (CR(0), "ascending", "before")  # order by cntrycode
        ])
        
        return result
