# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import gqe.lib
from gqe.relation import Relation


class Q13FilterOrdersRelation(Relation):
    """
    Q13 filter relation for the orders table.
    """

    def __init__(self, input: Relation):
        self.input = input

    def _to_cpp(self):
        return gqe.lib.q13_filter_orders(self.input._cpp)


class Q13GroupjoinBuildRelation(Relation):
    """
    Q13 build relation for the customer table.
    """

    def __init__(self, input: Relation, scale_factor: float):
        self.input = input
        self.scale_factor = scale_factor

    def _to_cpp(self):
        return gqe.lib.q13_groupjoin_build(self.input._cpp, self.scale_factor)


class Q13GroupjoinProbeRelation(Relation):
    """
    Q13 groupjoin probe relation for the orders table.
    """

    def __init__(self, groupjoin_build: Relation, orders: Relation):
        self.groupjoin_build = groupjoin_build
        self.orders = orders

    def _to_cpp(self):
        return gqe.lib.q13_groupjoin_probe(self.groupjoin_build._cpp, self.orders._cpp)


class Q13FusedFilterProbeRelation(Relation):
    """
    Q13 fused filter and groupjoin probe relation for the orders table.
    """

    def __init__(self, groupjoin_build: Relation, orders: Relation):
        self.groupjoin_build = groupjoin_build
        self.orders = orders

    def _to_cpp(self):
        return gqe.lib.q13_fused_filter_probe(self.groupjoin_build._cpp, self.orders._cpp)


class Q13GroupjoinRetrieveRelation(Relation):
    """
    Q13 groupjoin retrieve relation for the groupjoin result.
    """

    def __init__(self, input: Relation):
        self.input = input

    def _to_cpp(self):
        return gqe.lib.q13_groupjoin_retrieve(self.input._cpp)
