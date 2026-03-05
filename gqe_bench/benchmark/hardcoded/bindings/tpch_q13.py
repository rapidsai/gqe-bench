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

import gqe_bench.lib
from gqe_bench.relation import Relation


class Q13FilterOrdersRelation(Relation):
    """
    Q13 filter relation for the orders table.
    """

    def __init__(self, input: Relation):
        self.input = input

    def _to_cpp(self):
        return gqe_bench.lib.q13_filter_orders(self.input._cpp)


class Q13GroupjoinBuildRelation(Relation):
    """
    Q13 build relation for the customer table.
    """

    def __init__(self, input: Relation, scale_factor: float):
        self.input = input
        self.scale_factor = scale_factor

    def _to_cpp(self):
        return gqe_bench.lib.q13_groupjoin_build(self.input._cpp, self.scale_factor)


class Q13GroupjoinProbeRelation(Relation):
    """
    Q13 groupjoin probe relation for the orders table.
    """

    def __init__(self, groupjoin_build: Relation, orders: Relation):
        self.groupjoin_build = groupjoin_build
        self.orders = orders

    def _to_cpp(self):
        return gqe_bench.lib.q13_groupjoin_probe(self.groupjoin_build._cpp, self.orders._cpp)


class Q13FusedFilterProbeRelation(Relation):
    """
    Q13 fused filter and groupjoin probe relation for the orders table.
    """

    def __init__(self, groupjoin_build: Relation, orders: Relation):
        self.groupjoin_build = groupjoin_build
        self.orders = orders

    def _to_cpp(self):
        return gqe_bench.lib.q13_fused_filter_probe(self.groupjoin_build._cpp, self.orders._cpp)


class Q13GroupjoinRetrieveRelation(Relation):
    """
    Q13 groupjoin retrieve relation for the groupjoin result.
    """

    def __init__(self, input: Relation):
        self.input = input

    def _to_cpp(self):
        return gqe_bench.lib.q13_groupjoin_retrieve(self.input._cpp)
