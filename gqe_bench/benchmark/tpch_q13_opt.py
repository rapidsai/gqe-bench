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
from gqe_bench.benchmark.hardcoded.bindings.tpch_q13 import (
    Q13FilterOrdersRelation,
    Q13GroupjoinBuildRelation,
    Q13GroupjoinProbeRelation,
    Q13GroupjoinRetrieveRelation,
)
from gqe_bench.benchmark.query import Query
from gqe_bench.expression import ColumnReference as CR
from gqe_bench.table_definition import TPCHTableDefinitions


class tpch_q13_opt(Query):
    """
    Creates a TPC-H Q13 phyiscal query plan with hand-coded filter and
    groupjoin kernels.

    # NULL value optimization

    Q13 performs a left outer join followed by a group by. The intent is to
    return a row for each customer, even those with zero orders. "No orders" is
    represented as a NULL value, which the group by `COUNT(o_orderkey)`
    converts into a zero.

    However, in the groupjoin, the build inserts all customers. With no further
    adaptation, the retrieve emits all customers, thus naturally implements a
    left outer join. As there is no intermediate result between the join and
    group by, we don't need to handle NULL values. Instead, the groupjoin
    directly initializes all `c_count` values to zero.

    In effect, we don't need to read `o_orderkey`, because its only purpose is
    to represent non-NULL join results.

    # Transfer-compute overlap optimization

    The build, probe, and retrieve phases of the groupjoin must be in different
    execution stages to enable overlapping within a stage. This is because user
    defined function's task generator must emit multiple tasks for GQE to
    execute them concurrently. However, the intermediates must be concatenated,
    leading to a single task being emitted.

    In GQE, this is usually accomplished by inserting a pipeline breaker, which
    takes multiple tasks, and generates a new stage. For example, the broadcast
    join (with hash map caching) inserts a pipeline breaker after the build and
    after the probe. However, the user defined relation does not expose
    pipeline breakers. Therefore, the phases are explicitly exposed as separate
    relations, for which the task graph builder generates stages.

    # Limitations

    The query uses the `customer` table's cardinality to allocate a hash map
    with a corresponding size. The size is known from the schema catalog,
    because the hash map is built on the unfiltered `customer` table. However,
    as we don't have an API to the catalog, we instead workaround the missing
    size value by calculating it based on the scale factor.
    """

    def __init__(self, scale_factor: float):
        """
        Parameters:
           scale_factor (float): The TPC-H scale factor.
        """

        self.scale_factor = scale_factor

    def root_relation(self, table_defs: TPCHTableDefinitions):
        # Read customer table
        customer = read("customer", ["c_custkey"], None, table_defs)

        # Read orders table and filter rows where o_comment does not contain '%special%requests%'
        orders = read("orders", ["o_custkey", "o_comment"], None, table_defs)
        orders = Q13FilterOrdersRelation(orders)

        # Build the groupjoin hash map
        customer_build = Q13GroupjoinBuildRelation(customer, self.scale_factor)

        # Probe the groupjoin hash map
        groupjoin_intermediate = Q13GroupjoinProbeRelation(customer_build, orders)

        # Group by c_custkey and calculate count(o_orderkey) as c_count
        # After aggregation, the result contains: c_custkey, c_count
        grouped_customer_orders = Q13GroupjoinRetrieveRelation(groupjoin_intermediate)

        # Group by c_count and calculate count(*) as custdist
        # After aggregation, the result contains: c_count, custdist
        grouped_c_count = grouped_customer_orders.aggregate([CR(1)], [("count_all", CR(1))])

        # Sort by custdist DESC, c_count DESC
        result = grouped_c_count.sort(
            [(CR(1), "descending", "before"), (CR(0), "descending", "before")]
        )

        return result
