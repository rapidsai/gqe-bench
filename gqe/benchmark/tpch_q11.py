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
from gqe.expression import Literal
from gqe.benchmark.query import Query
from gqe.lib import UniqueKeysPolicy
from gqe.table_definition import TPCHTableDefinitions

import pandas as pd
from pandas.testing import assert_frame_equal


"""
select
        ps_partkey,
        sum(ps_supplycost * ps_availqty) as value
from
        partsupp,
        supplier,
        nation
where
        ps_suppkey = s_suppkey
        and s_nationkey = n_nationkey
        and n_name = 'GERMANY'
group by
        ps_partkey having
                sum(ps_supplycost * ps_availqty) > (
                        select
                                sum(ps_supplycost * ps_availqty) * [FRACTION]
                        from
                                partsupp,
                                supplier,
                                nation
                        where
                                ps_suppkey = s_suppkey
                                and s_nationkey = n_nationkey
                                and n_name = 'GERMANY'
                )
order by
        value desc

substitution parameters:
 - FRACTION is chosen as 0.0001 / SF
"""


class tpch_q11(Query):
    """
    Creates a TPC-H Q11 phyiscal query plan.

    The query definition depends on the scale factor, due to the FRACTION substitution parameter.
    """

    def __init__(self, scale_factor: int):
        """
        Parameters:
           scale_factor (int): The TPC-H scale factor.
        """

        self.scale_factor = scale_factor

    def root_relation(self, table_defs: TPCHTableDefinitions):
        # n_name = 'GERMANY'
        # After these operations, `nation` contains columns ["n_nationkey"]
        nation = read(
            "nation", ["n_nationkey", "n_name"], CR(1) == Literal("GERMANY"), table_defs
        )
        nation = nation.filter(CR(1) == Literal("GERMANY"), [0])

        # s_nationkey = n_nationkey
        # After these operations, `supplier` contains columns ["s_suppkey"]
        supplier = read("supplier", ["s_suppkey", "s_nationkey"], None, table_defs)
        supplier = supplier.broadcast_join(
            nation,
            CR(1) == CR(2),
            [0],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=False,
        )

        # ps_suppkey = s_suppkey
        # After these operations, `partsupp` contains columns
        # ["ps_partkey", "ps_supplycost", "ps_availqty"]
        partsupp = read(
            "partsupp",
            ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"],
            None,
            table_defs,
        )
        partsupp = partsupp.broadcast_join(
            supplier,
            CR(1) == CR(4),
            [0, 2, 3],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # Calculate sum(ps_supplycost * ps_availqty)
        sum_agg = partsupp.aggregate([], [("sum", CR(1) * CR(2))], perfect_hashing=True)

        # sum(ps_supplycost * ps_availqty) group by ps_partkey
        # After this operation, `partsupp` contains columns ["ps_partkey", value]
        partsupp = partsupp.aggregate(
            [CR(0)], [("sum", CR(1) * CR(2))], perfect_hashing=True
        )

        # having sum(ps_supplycost * ps_availqty) > [FRACTION] * ...
        # After this operation, `partsupp` contains columns ["ps_partkey", value]
        #
        # TPC-H Q11 has the substitution parameter `FRACTION`. Per the
        # specification "FRACTION is chosen as 0.0001 / SF". Therefore, adjust
        # `FRACTION` based on the scale factor.
        partsupp = partsupp.broadcast_join(
            sum_agg, CR(1) > (CR(2) * Literal(0.0001 / self.scale_factor)), [0, 1]
        )

        # order by value desc
        partsupp = partsupp.sort([(CR(1), "descending", "before")])

        return partsupp

    def validate_query(
        self, query_result: pd.DataFrame, reference: pd.DataFrame, atol: float
    ):
        """
        Validates the query result.

        TPC-H Q11 requires a custom validator because:

          1. The query doesn't fully sepecify the sort order of the result.
          There can be duplicate `value`s, that have different `ps_partkey`s.
          Thus, comparing results line-by-line can fail due to different line
          ordering than the reference result.

          2. Substituting the `DECIMAL` type with a floating-point type results
          in an unstable sort order. Rounding errors in `value` can lead to a
          different sort order than the reference result.

        Parameters:
          query_result (pd.DataFrame): The Q11 query result computed by GQE.
          reference (pd.DataFrame): The reference query result.
          atol (float): The floating-point tolerance for validation.
        """

        q_partkey_name = query_result.columns[0]
        q_value_name = query_result.columns[1]

        r_partkey_name = reference.columns[0]
        r_value_name = reference.columns[1]

        # Validate sort order.
        assert_frame_equal(
            query_result[[q_value_name]], reference[[r_value_name]], atol=atol
        )

        # Round to two fractional digits, as specified by TPC-H DECIMAL.
        query_result[q_value_name] = query_result[q_value_name].round(2)

        # Sort again after rounding. Specify a sort order on both columns.
        query_result = query_result.sort_values(
            by=[q_partkey_name, q_value_name], ascending=[False, True]
        )
        reference = reference.sort_values(
            by=[r_partkey_name, r_value_name], ascending=[False, True]
        )

        query_result.reset_index(drop=True, inplace=True)
        reference.reset_index(drop=True, inplace=True)

        # Validate row contents.
        assert_frame_equal(query_result, reference, atol=atol)
