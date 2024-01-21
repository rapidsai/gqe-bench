# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
In GQE, a **relation** takes zero or more input tables and evaluates to a single table. For example,
:class:`join <gqe.relation.JoinRelation>` is a relation that takes two input tables, namely the
left table and the right table, and evaluates to a table, the join result. Other relations includes
:class:`filter <gqe.relation.FilterRelation>`, :class:`aggregate <gqe.relation.AggregateRelation>`,
:class:`project <gqe.relation.ProjectRelation>`, etc. A relation is represented by an object of the
:class:`Relation <gqe.relation.Relation>` class.
"""

from __future__ import annotations  # Enable forward references for type annotations

from gqe.expression import Expression
from gqe.catalog import Catalog
import gqe.lib
from abc import ABC, abstractmethod
from functools import cached_property


class Relation(ABC):
    @cached_property
    def _cpp(self) -> gqe.lib.Relation:
        """
        Convert `self` into the equivalent C++ Relation object.

        Note that this method must be cached to allow multiple relations share a common child
        relation.
        """
        return self._to_cpp()

    @abstractmethod
    def _to_cpp(self) -> gqe.lib.Relation:
        pass

    def filter(self, condition: Expression) -> Relation:
        """
        Select a subset of rows based on the condition.

        :param condition: A row will be included in the output table if and only if the condition
            evaluates to `True`.
        """
        return FilterRelation(self, condition)

    def join(self, other: Relation, condition: Expression, projection_indices: list[int],
             type: str = "inner") -> Relation:
        """
        Join with another table.

        :param other: The other table to be joined with the current table.
        :param condition: `condition` determines when a left row matches a right row. Note that the
            column index of `other` starts after `self`, so if `self` has `n` column, the `i`-th
            column of `other` is referred as `ColumnReference(n+i)`.
        :param projection_indices: Columns to materialize for the output. The output table will have
            the same number of columns as `len(projection_indices)`.
        :param type: Type of the join. Acceptable values are `"inner"`, `"left"`, `"left_semi"`,
            `"left_anti"` and `"full"`.
        """
        return JoinRelation(self, other, condition, projection_indices, type)

    def aggregate(self, keys: list[Expression], measures: list[tuple[str, Expression]]) -> Relation:
        """
        Groups the table on zero or more sets of grouping keys and applies reduction within the
        groups.

        The output table contains key columns followed by measure columns, so the number of columns
        of the output is `len(keys) + len(measures)`.

        :param keys: Grouping keys. If the list is empty, the output table has one row, namely the
            reduction results of all rows in the input table. If the list is not empty, the
            aggregate relation behaves like a groupby.
        :param measures: A list of (op, expression) pairs, representing the reduction operations
            within the groups. `op` can be `"sum"` or `"avg"`.
        """
        return AggregateRelation(self, keys, measures)

    def project(self, out_exprs: list[Expression]) -> Relation:
        """
        Evaluates expressions on the table.

        The output table contains the same number of columns as `len(out_exprs)`.

        :param out_exprs: Output expressions to be evaluated.
        """
        return ProjectRelation(self, out_exprs)


class ReadRelation(Relation):
    """
    A read relation reads a table from the catalog. It does not take any input tables.
    """
    def __init__(self, catalog: Catalog, table: str, columns: list[str]):
        self.table = table
        self.columns = columns
        self.catalog = catalog

    def _to_cpp(self):
        return gqe.lib.read(self.catalog._catalog, self.table, self.columns)


def read(catalog: Catalog, table: str, columns: list[str]) -> Relation:
    """
    Factory for constructing a read relation.

    :param catalog: Catalog to read the table from.
    :param table: Name of the table to load.
    :param columns: Name of the columns to load.
    """
    return ReadRelation(catalog, table, columns)


class FilterRelation(Relation):
    """
    A filter relation selects a subset of rows from the input table based on the filter condition.
    """
    def __init__(self, input: Relation, condition: Expression):
        self.input = input
        self.condition = condition

    def _to_cpp(self):
        return gqe.lib.filter(self.input._cpp, self.condition._cpp)


_join_type_to_cpp: dict[str, gqe.lib.JoinType] = {
    "inner": gqe.lib.JoinType.inner,
    "left": gqe.lib.JoinType.left,
    "left_semi": gqe.lib.JoinType.left_semi,
    "left_anti": gqe.lib.JoinType.left_anti,
    "full": gqe.lib.JoinType.full,
}


class JoinRelation(Relation):
    """
    A join relation performs a join operation on the two input tables in relational algebra, and
    output the join result.
    """
    def __init__(self, left: Relation, right: Relation, condition: Expression,
                 projection_indices: list[int],
                 type: str = "inner"):
        self.left = left
        self.right = right
        self.condition = condition
        self.projection_indices = projection_indices

        if type not in _join_type_to_cpp:
            raise ValueError(f"Unknown join type: {type}")
        else:
            self.type = type

    def _to_cpp(self):
        return gqe.lib.join(
            self.left._cpp, self.right._cpp, self.condition._cpp,
            _join_type_to_cpp[self.type],
            self.projection_indices)


_aggregation_kind_to_cpp: dict[str, gqe.lib.AggregationKind] = {
    "sum": gqe.lib.AggregationKind.sum,
    "avg": gqe.lib.AggregationKind.avg,
}


class AggregateRelation(Relation):
    """
    An aggregate relation groups the input table on zero or more sets of grouping keys and applies
    reduction within the groups.
    """
    def __init__(self, input: Relation, keys: list[Expression],
                 measures: list[tuple[str, Expression]]):
        self.input = input
        self.keys = keys

        for (kind, _) in measures:
            if kind not in _aggregation_kind_to_cpp:
                raise ValueError(f"Unknown aggregation kind: {kind}")

        self.measures = measures

    def _to_cpp(self):
        return gqe.lib.aggregate(
            self.input._cpp,
            [key._cpp for key in self.keys],
            [(_aggregation_kind_to_cpp[kind], expr._cpp) for (kind, expr) in self.measures])


class ProjectRelation(Relation):
    """
    A project relation evaluates output expressions on the input table.
    """
    def __init__(self, input: Relation, out_exprs: list[Expression]):
        self.input = input
        self.out_exprs = out_exprs

    def _to_cpp(self):
        return gqe.lib.project(
            self.input._cpp,
            [expr._cpp for expr in self.out_exprs])
