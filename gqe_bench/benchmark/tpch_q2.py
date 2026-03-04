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
from gqe_bench.expression import LikeExpr, Literal
from gqe_bench.lib import UniqueKeysPolicy
from gqe_bench.table_definition import TPCHTableDefinitions

"""
select
        s_acctbal,
        s_name,
        n_name,
        p_partkey,
        p_mfgr,
        s_address,
        s_phone,
        s_comment
from
        part,
        supplier,
        partsupp,
        nation,
        region
where
        p_partkey = ps_partkey
        and s_suppkey = ps_suppkey
        and p_size = 15
        and p_type like '%BRASS'
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = 'EUROPE'
        and ps_supplycost = (
                select
                        min(ps_supplycost)
                from
                        partsupp,
                        supplier,
                        nation,
                        region
                where
                        p_partkey = ps_partkey
                        and s_suppkey = ps_suppkey
                        and s_nationkey = n_nationkey
                        and n_regionkey = r_regionkey
                        and r_name = 'EUROPE'
        )
order by
        s_acctbal desc,
        n_name,
        s_name,
        p_partkey
limit
        100
"""


class tpch_q2(Query):
    def root_relation(self, table_defs: TPCHTableDefinitions):
        # After these operations, `part` contains columns ["p_partkey", "p_mfgr"]
        # It filters out many rows. The selectivity is only ~0.4%.
        part = read(
            "part",
            ["p_partkey", "p_size", "p_type", "p_mfgr"],
            (CR(5) == Literal(15)),
            table_defs,
        )
        part = part.filter((CR(1) == Literal(15)) & LikeExpr(CR(2), "%BRASS"), [0, 3])

        # After these operations, `region` contains columns ["r_regionkey"]
        region = read(
            "region",
            ["r_regionkey", "r_name"],
            (CR(1) == Literal("EUROPE")),
            table_defs,
        )
        region = region.filter(CR(1) == Literal("EUROPE"), [0])

        # After these operations, `nation` contains columns ["n_nationkey", "n_name"]
        nation = read("nation", ["n_nationkey", "n_regionkey", "n_name"], None, table_defs)
        nation = nation.broadcast_join(
            region,
            CR(1) == CR(3),
            [0, 2],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # After these operations, `supplier` contains columns
        # ["s_suppkey", "s_acctbal", "s_name", "s_address", "s_phone", "s_comment", "n_name"]
        supplier = read(
            "supplier",
            [
                "s_suppkey",
                "s_nationkey",
                "s_acctbal",
                "s_name",
                "s_address",
                "s_phone",
                "s_comment",
            ],
            None,
            table_defs,
        )
        supplier = supplier.broadcast_join(
            nation,
            CR(1) == CR(7),
            [0, 2, 3, 4, 5, 6, 8],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # Because p_partkey is a primary key, we can push the filter into the subquery
        # After these operations, `partsupp` contains columns
        # ["ps_partkey", "ps_suppkey", "ps_supplycost", "p_mfgr"]
        partsupp = read("partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost"], None, table_defs)
        partsupp = partsupp.broadcast_join(
            part,
            CR(0) == CR(3),
            [0, 1, 2, 4],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # After this operation, `partsupp` contains columns
        # ["ps_partkey", "ps_supplycost", "p_mfgr", "s_acctbal",
        #  "s_name", "s_address", "s_phone", "s_comment", "n_name"]
        partsupp = partsupp.broadcast_join(
            supplier,
            CR(1) == CR(4),
            [0, 2, 3, 5, 6, 7, 8, 9, 10],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # After these operations, `agg_result` contains columns ["ps_partkey", min(ps_supplycost)]
        agg_result = partsupp.aggregate([CR(0)], [("min", CR(1))], perfect_hashing=True)

        # After this operation, `partsupp` contains columns
        # ["s_acctbal", "s_name", "n_name", "p_partkey",
        #  "p_mfgr", "s_address", "s_phone", "s_comment"]
        partsupp = partsupp.broadcast_join(
            agg_result,
            (CR(0) == CR(9)) & (CR(1) == CR(10)),
            [3, 4, 8, 0, 2, 5, 6, 7],
            unique_keys_policy=UniqueKeysPolicy.right,
            perfect_hashing=True,
        )

        # order by s_acctbal desc, n_name, s_name, p_partkey
        partsupp = partsupp.sort(
            [
                (CR(0), "descending", "before"),
                (CR(2), "ascending", "before"),
                (CR(1), "ascending", "before"),
                (CR(3), "ascending", "before"),
            ]
        ).fetch(0, 100)

        return partsupp
