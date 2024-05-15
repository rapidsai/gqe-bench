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


'''
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
                                sum(ps_supplycost * ps_availqty) * 0.0001
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
'''


class tpch_q11(Query):
    def root_relation(self):
        # n_name = 'GERMANY'
        # After these operations, `nation` contains columns ["n_nationkey"]
        nation = read("nation", ["n_nationkey", "n_name"])
        nation = nation.filter(CR(1) == Literal("GERMANY"), [0])

        # s_nationkey = n_nationkey
        # After these operations, `supplier` contains columns ["s_suppkey"]
        supplier = read("supplier", ["s_suppkey", "s_nationkey"])
        supplier = supplier.broadcast_join(nation, CR(1) == CR(2), [0])

        # ps_suppkey = s_suppkey
        # After these operations, `partsupp` contains columns
        # ["ps_partkey", "ps_supplycost", "ps_availqty"]
        partsupp = read("partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"])
        partsupp = partsupp.broadcast_join(supplier, CR(1) == CR(4), [0, 2, 3])

        # Calculate sum(ps_supplycost * ps_availqty) * 0.0001
        # FIXME: for some reason CR(1) * CR(2) * Literal(0.0001) fails
        sum_agg = partsupp.aggregate([], [("sum", CR(1) * (CR(2) * Literal(0.0001)))])

        # sum(ps_supplycost * ps_availqty) group by ps_partkey
        # After this operation, `partsupp` contains columns ["ps_partkey", value]
        partsupp = partsupp.aggregate([CR(0)], [("sum", CR(1) * CR(2))])

        # having sum(ps_supplycost * ps_availqty) > ...
        # After this operation, `partsupp` contains columns ["ps_partkey", value]
        partsupp = partsupp.broadcast_join(sum_agg, CR(1) > CR(2), [0, 1])

        # order by value desc
        partsupp = partsupp.sort([(CR(1), "descending", "before")])

        return partsupp
