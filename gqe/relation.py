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
:class:`broadcast join <gqe.relation.BroadcastJoinRelation>` is a relation that takes two input
tables, namely the left table and the right table, and evaluates to a table, the join result. Other
relations includes :class:`filter <gqe.relation.FilterRelation>`,
:class:`aggregate <gqe.relation.AggregateRelation>`,
:class:`project <gqe.relation.ProjectRelation>`, etc. A relation is represented by an object of the
:class:`Relation <gqe.relation.Relation>` class.
"""

from __future__ import annotations  # Enable forward references for type annotations

from gqe.expression import Expression
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

    def filter(self, condition: Expression, projection_indices: list[int]) -> Relation:
        """
        Select a subset of rows based on the condition.

        :param condition: A row will be included in the output table if and only if the condition
            evaluates to `True`.
        """
        return FilterRelation(self, condition, projection_indices)

    def broadcast_join(
        self,
        right_table: Relation,
        condition: Expression,
        projection_indices: list[int],
        type: str = "inner",
        broadcast_left: bool = False,
        unique_keys_policy: gqe.lib.UniqueKeysPolicy = gqe.lib.UniqueKeysPolicy.none,
        perfect_hashing: bool = False
    ) -> Relation:
        """
        Join with another table by broadcasting

        :param right_table: The right-hand side of the join.
        :param condition: `condition` determines when a left row matches a right row. Note that the
            column index of `other` starts after `self`, so if `self` has `n` column, the `i`-th
            column of `other` is referred as `ColumnReference(n+i)`.
        :param projection_indices: Columns to materialize for the output. The output table will have
            the same number of columns as `len(projection_indices)`.
        :param type: Type of the join. Acceptable values are `"inner"`, `"left"`, `"left_semi"`,
            `"left_anti"` and `"full"`.
        :param broadcast_left: Whether to broadcast the left table.
        """
        return BroadcastJoinRelation(
            self, right_table, condition, projection_indices, type, broadcast_left, unique_keys_policy, perfect_hashing
        )

    def aggregate(
        self, keys: list[Expression], measures: list[tuple[str, Expression]], condition: Expression = None
    ) -> Relation:
        """
        Groups the table on zero or more sets of grouping keys and applies reduction within the
        groups.

        The output table contains key columns followed by measure columns, so the number of columns
        of the output is `len(keys) + len(measures)`.

        :param keys: Grouping keys. If the list is empty, the output table has one row, namely the
            reduction results of all rows in the input table. If the list is not empty, the
            aggregate relation behaves like a groupby.
        :param measures: A list of (op, expression) pairs, representing the reduction operations
            within the groups.
        :param condition: An optional expression to filter rows before aggregation. Only rows
            where the condition evaluates to `True` are included. If not provided, all rows
            are aggregated.
        """
        return AggregateRelation(self, keys, measures, condition)

    def project(self, out_exprs: list[Expression]) -> Relation:
        """
        Evaluates expressions on the table.

        The output table contains the same number of columns as `len(out_exprs)`.

        :param out_exprs: Output expressions to be evaluated.
        """
        return ProjectRelation(self, out_exprs)

    def sort(self, keys: list[tuple[Expression, str, str]]) -> Relation:
        """
        Reorder the table according to a set of keys.

        :param keys: A list of tuples `(key, column_order, null_precedence)`, from the most
            significant to the least significant, where `key` is evaluated on the input table,
            `column_order` can be "ascending" or "descending", and `null_precedence` can be "after"
            or "before".
        """
        return SortRelation(self, keys)

    def fetch(self, offset, count) -> Relation:
        """
        Select a rows within the start offset and the end offset.

        Note that if `offset + count` goes past the end of the table, this relation would output
        rows from `offset` to the end of the table.

        :param offset: Start offset to output.
        :param count: Number of rows to output.
        """
        return FetchRelation(self, offset, count)

    def union_all(self, other: Relation) -> Relation:
        """
        Append rows from another table.

        :param other: The other table to be appended at the end of this table.
        """
        return UnionAllRelation(self, other)


class ReadRelation(Relation):
    """
    A read relation reads a table from the catalog. It does not take any input tables.
    """

    def __init__(self, table: str, columns: list[str], partial_filter: Expression = None):
        self.table = table
        self.columns = columns
        self.partial_filter = partial_filter

    def _to_cpp(self):
        return gqe.lib.read(self.table, self.columns, self.partial_filter._cpp if self.partial_filter else None)


def read(table: str, columns: list[str], partial_filter: Expression = None) -> Relation:
    """
    Factory for constructing a read relation.

    :param table: Name of the table to load.
    :param columns: Name of the columns to load.
    """
    return ReadRelation(table, columns, partial_filter)


class FilterRelation(Relation):
    """
    A filter relation selects a subset of rows from the input table based on the filter condition.
    """

    def __init__(
        self, input: Relation, condition: Expression, projection_indices: list[int]
    ):
        self.input = input
        self.condition = condition
        self.projection_indices = projection_indices

    def _to_cpp(self):
        return gqe.lib.filter(
            self.input._cpp, self.condition._cpp, self.projection_indices
        )


_join_type_to_cpp: dict[str, gqe.lib.JoinType] = {
    "inner": gqe.lib.JoinType.inner,
    "left": gqe.lib.JoinType.left,
    "left_semi": gqe.lib.JoinType.left_semi,
    "left_anti": gqe.lib.JoinType.left_anti,
    "full": gqe.lib.JoinType.full,
}


class BroadcastJoinRelation(Relation):
    """
    A join relation performs a join operation on the two input tables in relational algebra, and
    output the join result.
    """

    def __init__(
        self,
        left_table: Relation,
        right_table: Relation,
        condition: Expression,
        projection_indices: list[int],
        type: str = "inner",
        broadcast_left: bool = False,
        unique_keys_policy: gqe.lib.UniqueKeysPolicy = gqe.lib.UniqueKeysPolicy.none,
        perfect_hashing: bool = False
    ):
        self.left_table = left_table
        self.right_table = right_table
        self.condition = condition
        self.projection_indices = projection_indices
        self.broadcast_left = broadcast_left
        self.unique_keys_policy = unique_keys_policy
        self.perfect_hashing = perfect_hashing

        if type not in _join_type_to_cpp:
            raise ValueError(f"Unknown join type: {type}")
        else:
            self.type = type

    def _to_cpp(self):
        return gqe.lib.broadcast_join(
            self.left_table._cpp,
            self.right_table._cpp,
            self.condition._cpp,
            _join_type_to_cpp[self.type],
            self.projection_indices,
            self.broadcast_left,
            self.unique_keys_policy,
            self.perfect_hashing,
        )


_aggregation_kind_to_cpp: dict[str, gqe.lib.AggregationKind] = {
    "sum": gqe.lib.AggregationKind.sum,
    "avg": gqe.lib.AggregationKind.avg,
    "count_all": gqe.lib.AggregationKind.count_all,
    "count_valid": gqe.lib.AggregationKind.count_valid,
    "min": gqe.lib.AggregationKind.min,
    "max": gqe.lib.AggregationKind.max,
}


class AggregateRelation(Relation):
    """
    An aggregate relation groups the input table on zero or more sets of grouping keys and applies
    reduction within the groups.
    """

    def __init__(
        self,
        input: Relation,
        keys: list[Expression],
        measures: list[tuple[str, Expression]],
        condition: Expression = None,
    ):
        self.input = input

        for key in keys:
            if not isinstance(key, Expression):
                raise TypeError("Aggregate keys are not expressions")

        self.keys = keys

        for kind, expr in measures:
            if kind not in _aggregation_kind_to_cpp:
                raise ValueError(f"Unknown aggregation kind: {kind}")
            if not isinstance(expr, Expression):
                raise TypeError("Aggregate reductions are not expressions")

        self.measures = measures
        self.condition = condition

    def _to_cpp(self):
        return gqe.lib.aggregate(
            self.input._cpp,
            [key._cpp for key in self.keys],
            [
                (_aggregation_kind_to_cpp[kind], expr._cpp)
                for (kind, expr) in self.measures
            ],
            self.condition._cpp if self.condition else None
        )


class ProjectRelation(Relation):
    """
    A project relation evaluates output expressions on the input table.
    """

    def __init__(self, input: Relation, out_exprs: list[Expression]):
        self.input = input
        self.out_exprs = out_exprs

    def _to_cpp(self):
        return gqe.lib.project(self.input._cpp, [expr._cpp for expr in self.out_exprs])


_order_to_cpp: dict[str, gqe.lib.Order] = {
    "ascending": gqe.lib.Order.ascending,
    "descending": gqe.lib.Order.descending,
}

_null_order_to_cpp: dict[str, gqe.lib.NullOrder] = {
    "after": gqe.lib.NullOrder.after,
    "before": gqe.lib.NullOrder.before,
}


class SortRelation(Relation):
    """
    A sort relation reorders the rows of the input table according to a set of keys.
    """

    def __init__(self, input: Relation, keys: list[tuple[Expression, str, str]]):
        self.input = input

        for expr, order, null_order in keys:
            if not isinstance(expr, Expression):
                raise TypeError("Sort keys are not expressions")
            if order not in _order_to_cpp:
                raise ValueError(f"Unknown sort column order: {order}")
            if null_order not in _null_order_to_cpp:
                raise ValueError(f"Unknown sort null precedence: {null_order}")

        self.keys = keys

    def _to_cpp(self) -> gqe.lib.Relation:
        exprs = []
        orders = []
        null_orders = []

        for expr, order, null_order in self.keys:
            exprs.append(expr)
            orders.append(_order_to_cpp[order])
            null_orders.append(_null_order_to_cpp[null_order])

        return gqe.lib.sort(
            self.input._cpp, orders, null_orders, [expr._cpp for expr in exprs]
        )


class FetchRelation(Relation):
    """
    A fetch relation outputs rows within the start offset and the end offset.
    """

    def __init__(self, input: Relation, offset: int, count: int):
        self.input = input
        self.offset = offset
        self.count = count

    def _to_cpp(self) -> gqe.lib.Relation:
        return gqe.lib.fetch(self.input._cpp, self.offset, self.count)

class UnionAllRelation(Relation):
    """
    A union-all relation combines the rows of two input tables, retaining any duplicate rows.
    """

    def __init__(self, lhs: Relation, rhs: Relation):
        self.lhs = lhs
        self.rhs = rhs

    def _to_cpp(self) -> gqe.lib.Relation:
        return gqe.lib.union_all(self.lhs._cpp, self.rhs._cpp)
