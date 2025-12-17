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
from gqe.benchmark.query import Query
from gqe.expression import ColumnReference as CR
from gqe.expression import Literal, SubstrExpr
from gqe.table_definition import TPCHTableDefinitions

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


class tpch_q22(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # customer: c_custkey, substring(c_phone, 0, 2), c_acctbal
        customer = read("customer", ["c_custkey", "c_phone", "c_acctbal"], None, table_defs)
        customer = customer.project([CR(0), SubstrExpr(CR(1), 0, 2), CR(2)])

        # List of country codes to filter by
        country_codes = ["13", "31", "23", "29", "30", "18", "17"]

        # Create a country code filter condition
        # SQL substring(c_phone from 1 for 2) converts to substr(0, 2) in 0-indexed systems
        filter_condition = None
        for code in country_codes:
            condition = CR(1) == Literal(code)
            if filter_condition is None:
                filter_condition = condition
            else:
                filter_condition = filter_condition | condition

        # Filter customers with matching country codes
        # filtered_customers: c_custkey, substring(c_phone, 0, 2), c_acctbal
        filtered_customers = customer.filter(filter_condition, [0, 1, 2])

        # Calculate average account balance for positive balance customers with matching country codes
        pos_balance_customers = filtered_customers.filter(CR(2) > Literal(0.0), [2])
        avg_acctbal = pos_balance_customers.aggregate([], [("avg", CR(0))], perfect_hashing=False)

        # Filter customers with account balance > average
        high_balance_customers = filtered_customers.broadcast_join(
            avg_acctbal, CR(2) > CR(3), [0, 1, 2], "left_semi"
        )

        # Find customers with no orders using left anti join
        orders = read("orders", ["o_custkey"], None, table_defs)
        customers_no_orders = high_balance_customers.broadcast_join(
            orders, CR(0) == CR(3), [1, 2], "left_anti", broadcast_left=True
        )

        # Group by country code and calculate aggregates
        # result: cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal
        result = customers_no_orders.aggregate(
            [CR(0)],  # Group by cntrycode
            [
                ("count_all", CR(0)),  # count(*) as numcust
                ("sum", CR(1)),  # sum(c_acctbal) as totacctbal
            ],
            perfect_hashing=False,
        )

        # Order by country code
        result = result.sort([(CR(0), "ascending", "before")])  # order by cntrycode

        return result
