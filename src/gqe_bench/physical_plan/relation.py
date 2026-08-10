# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relation tree for physical plan construction.

Serializes via grpc.protos()-generated message classes from
physical_plan.proto. Provides a method-chaining API for building relation
trees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum

from gqe_bench.physical_plan import proto
from gqe_bench.physical_plan.expression import DataType, Expression


class JoinType(IntEnum):
    """SQL join semantics."""

    INNER = 0
    LEFT = 1
    LEFT_SEMI = 2
    LEFT_ANTI = 3
    FULL = 4
    SINGLE = 5


class UniqueKeysPolicy(IntEnum):
    """Which side(s) of a join can be assumed to have unique keys."""

    NONE = 0
    RIGHT = 1
    LEFT = 2
    EITHER = 3


class BroadcastPolicy(IntEnum):
    """Which side of a broadcast join is the broadcast side."""

    RIGHT = 0
    LEFT = 1


class AggregationKind(IntEnum):
    """Aggregation functions.

    Values mirror ``cudf::aggregation::Kind`` in cudf/aggregation.hpp;
    the proto carries this as int32 reinterpret-cast on the server side.
    """

    SUM = 0
    SUM_WITH_OVERFLOW = 1
    PRODUCT = 2
    MIN = 3
    MAX = 4
    COUNT_VALID = 5
    COUNT_ALL = 6
    ANY = 7
    ALL = 8
    SUM_OF_SQUARES = 9
    MEAN = 10
    NUNIQUE = 18


class Order(IntEnum):
    """Sort direction."""

    ASCENDING = 0
    DESCENDING = 1


class NullOrder(IntEnum):
    """Whether NULLs sort before or after non-NULL values."""

    BEFORE = 0
    AFTER = 1


_JOIN_TYPE_MAP: dict[str, JoinType] = {
    "inner": JoinType.INNER,
    "left": JoinType.LEFT,
    "left_semi": JoinType.LEFT_SEMI,
    "left_anti": JoinType.LEFT_ANTI,
    "full": JoinType.FULL,
}

_AGG_KIND_MAP: dict[str, AggregationKind] = {
    "sum": AggregationKind.SUM,
    "avg": AggregationKind.MEAN,
    "count_all": AggregationKind.COUNT_ALL,
    "count_valid": AggregationKind.COUNT_VALID,
    "min": AggregationKind.MIN,
    "max": AggregationKind.MAX,
}

_ORDER_MAP: dict[str, Order] = {
    "ascending": Order.ASCENDING,
    "descending": Order.DESCENDING,
}

_NULL_ORDER_MAP: dict[str, NullOrder] = {
    "after": NullOrder.AFTER,
    "before": NullOrder.BEFORE,
}


class SerializationContext:
    """Ids assigned to the relations of one plan, keyed on ``id()``."""

    def __init__(self) -> None:
        self._assigned: dict[int, int] = {}
        self._next: int = 1  # 0 is the unset sentinel for PhysicalRelation.node_id

    def emit(self, relation: Relation) -> proto.physical_plan.PhysicalRelation:
        """Serialize ``relation``, or emit a ReferenceRel if already serialized."""
        assigned = self._assigned.get(id(relation))
        if assigned is not None:
            msg = proto.physical_plan.PhysicalRelation()
            msg.reference_rel.node_id = assigned
            return msg

        node_id = self._next
        self._next += 1
        self._assigned[id(relation)] = node_id

        msg = relation.to_proto(self)
        msg.node_id = node_id
        return msg


class Relation(ABC):
    """Abstract base for physical-plan relation nodes.

    Subclasses implement ``to_proto`` to emit the matching
    ``PhysicalRelation`` oneof. Builder methods on this base compose
    subclasses into a DAG.
    """

    @abstractmethod
    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        """Build a PhysicalRelation protobuf message.

        Emit child relations with ``ctx.emit(child)`` rather than
        ``child.to_proto``, or sharing is lost.
        """

    def serialize(self) -> bytes:
        """Serialize as a gqe.proto.PhysicalRelation message."""
        return SerializationContext().emit(self).SerializeToString()

    def filter(
        self,
        condition: Expression,
        projection_indices: list[int],
    ) -> Relation:
        """Apply a boolean filter."""
        return FilterRelation(self, condition, projection_indices)

    def broadcast_join(
        self,
        right_table: Relation,
        condition: Expression,
        projection_indices: list[int],
        join_type: str = "inner",
        broadcast_left: bool = False,
        unique_keys_policy: UniqueKeysPolicy = UniqueKeysPolicy.NONE,
        perfect_hashing: bool = False,
        left_filter: Expression | None = None,
        right_filter: Expression | None = None,
    ) -> Relation:
        """Broadcast-join this relation with ``right_table``."""
        return BroadcastJoinRelation(
            self,
            right_table,
            condition,
            projection_indices,
            join_type,
            broadcast_left,
            unique_keys_policy,
            perfect_hashing,
            left_filter,
            right_filter,
        )

    def shuffle_join(
        self,
        right_table: Relation,
        condition: Expression,
        projection_indices: list[int],
        join_type: str = "inner",
        unique_keys_policy: UniqueKeysPolicy = UniqueKeysPolicy.NONE,
        perfect_hashing: bool = False,
    ) -> Relation:
        """Shuffle-join this relation with ``right_table``."""
        return ShuffleJoinRelation(
            self,
            right_table,
            condition,
            projection_indices,
            join_type,
            unique_keys_policy,
            perfect_hashing,
        )

    def shuffle(self, shuffle_cols: list[Expression]) -> Relation:
        """Shuffle (repartition) by ``shuffle_cols``."""
        return ShuffleRelation(self, shuffle_cols)

    def aggregate(
        self,
        keys: list[Expression],
        measures: list[tuple[str, Expression]],
        condition: Expression | None = None,
        perfect_hashing: bool = False,
    ) -> Relation:
        """Group-by + aggregate."""
        return AggregateRelation(self, keys, measures, condition, perfect_hashing)

    def project(self, out_exprs: list[Expression]) -> Relation:
        """Compute output expressions."""
        return ProjectRelation(self, out_exprs)

    def sort(self, keys: list[tuple[Expression, str, str]]) -> Relation:
        """Sort by ``(expression, order, null_precedence)`` keys."""
        return SortRelation(self, keys)

    def fetch(self, offset: int, count: int) -> Relation:
        """Apply limit/offset."""
        return FetchRelation(self, offset, count)

    def union_all(self, other: Relation) -> Relation:
        """Concatenate this relation with ``other``."""
        return UnionAllRelation(self, other)


class ReadRelation(Relation):
    """Source table read with optional partial filter pushdown.

    ``data_types`` is index-aligned with ``columns`` (one type per
    selected column, in the same order).
    """

    def __init__(
        self,
        table: str,
        columns: list[str],
        data_types: list[DataType],
        partial_filter: Expression | None = None,
    ) -> None:
        self.table = table
        self.columns = columns
        self.data_types = data_types
        self.partial_filter = partial_filter

    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        msg = proto.physical_plan.PhysicalRelation()
        read = msg.read
        read.table_name = self.table
        read.column_names.extend(self.columns)
        for dt in self.data_types:
            read.data_types.add().CopyFrom(dt.to_proto())
        if self.partial_filter is not None:
            read.partial_filter.CopyFrom(self.partial_filter.to_proto())
        return msg


class FilterRelation(Relation):
    """Boolean filter with optional projection."""

    def __init__(
        self,
        input: Relation,
        condition: Expression,
        projection_indices: list[int],
    ) -> None:
        self.input = input
        self.condition = condition
        self.projection_indices = projection_indices

    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        msg = proto.physical_plan.PhysicalRelation()
        f = msg.filter
        f.child.CopyFrom(ctx.emit(self.input))
        f.condition.CopyFrom(self.condition.to_proto())
        f.projection_indices.extend(self.projection_indices)
        return msg


class BroadcastJoinRelation(Relation):
    """Join where one side is broadcast to every worker.

    ``broadcast_left`` selects which side. ``__init__`` raises ValueError
    if ``join_type`` isn't one of the keys in ``_JOIN_TYPE_MAP``.
    """

    def __init__(
        self,
        left_table: Relation,
        right_table: Relation,
        condition: Expression,
        projection_indices: list[int],
        join_type: str = "inner",
        broadcast_left: bool = False,
        unique_keys_policy: UniqueKeysPolicy = UniqueKeysPolicy.NONE,
        perfect_hashing: bool = False,
        left_filter: Expression | None = None,
        right_filter: Expression | None = None,
    ) -> None:
        if join_type not in _JOIN_TYPE_MAP:
            raise ValueError(f"Unknown join type: {join_type}")
        self.left_table = left_table
        self.right_table = right_table
        self.condition = condition
        self.projection_indices = projection_indices
        self.join_type = join_type
        self.broadcast_left = broadcast_left
        self.unique_keys_policy = unique_keys_policy
        self.perfect_hashing = perfect_hashing
        self.left_filter = left_filter
        self.right_filter = right_filter

    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        policy = BroadcastPolicy.LEFT if self.broadcast_left else BroadcastPolicy.RIGHT
        msg = proto.physical_plan.PhysicalRelation()
        j = msg.broadcast_join
        j.left.CopyFrom(ctx.emit(self.left_table))
        j.right.CopyFrom(ctx.emit(self.right_table))
        j.join_type = _JOIN_TYPE_MAP[self.join_type]
        j.condition.CopyFrom(self.condition.to_proto())
        j.projection_indices.extend(self.projection_indices)
        j.broadcast_policy = policy
        j.unique_keys_policy = self.unique_keys_policy
        j.perfect_hashing = self.perfect_hashing
        if self.left_filter is not None:
            j.left_filter_condition.CopyFrom(self.left_filter.to_proto())
        if self.right_filter is not None:
            j.right_filter_condition.CopyFrom(self.right_filter.to_proto())
        return msg


class ShuffleJoinRelation(Relation):
    """Join where both inputs are shuffled by their join keys.

    ``__init__`` raises ValueError if ``join_type`` isn't one of the keys
    in ``_JOIN_TYPE_MAP``.
    """

    def __init__(
        self,
        left_table: Relation,
        right_table: Relation,
        condition: Expression,
        projection_indices: list[int],
        join_type: str = "inner",
        unique_keys_policy: UniqueKeysPolicy = UniqueKeysPolicy.NONE,
        perfect_hashing: bool = False,
    ) -> None:
        if join_type not in _JOIN_TYPE_MAP:
            raise ValueError(f"Unknown join type: {join_type}")
        self.left_table = left_table
        self.right_table = right_table
        self.condition = condition
        self.projection_indices = projection_indices
        self.join_type = join_type
        self.unique_keys_policy = unique_keys_policy
        self.perfect_hashing = perfect_hashing

    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        msg = proto.physical_plan.PhysicalRelation()
        j = msg.shuffle_join
        j.left.CopyFrom(ctx.emit(self.left_table))
        j.right.CopyFrom(ctx.emit(self.right_table))
        j.join_type = _JOIN_TYPE_MAP[self.join_type]
        j.condition.CopyFrom(self.condition.to_proto())
        j.projection_indices.extend(self.projection_indices)
        j.unique_keys_policy = self.unique_keys_policy
        j.perfect_hashing = self.perfect_hashing
        return msg


class ProjectRelation(Relation):
    """Compute output expressions over an input relation."""

    def __init__(self, input: Relation, out_exprs: list[Expression]) -> None:
        self.input = input
        self.out_exprs = out_exprs

    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        msg = proto.physical_plan.PhysicalRelation()
        p = msg.project
        p.child.CopyFrom(ctx.emit(self.input))
        for expr in self.out_exprs:
            p.output_expressions.add().CopyFrom(expr.to_proto())
        return msg


class AggregateRelation(Relation):
    """Group-by + aggregate. ``measures`` are ``(kind, expression)`` pairs.

    ``__init__`` raises ValueError if any ``measure[0]`` isn't a known
    aggregation kind.
    """

    def __init__(
        self,
        input: Relation,
        keys: list[Expression],
        measures: list[tuple[str, Expression]],
        condition: Expression | None = None,
        perfect_hashing: bool = False,
    ) -> None:
        for kind, _ in measures:
            if kind not in _AGG_KIND_MAP:
                raise ValueError(f"Unknown aggregation kind: {kind}")
        self.input = input
        self.keys = keys
        self.measures = measures
        self.condition = condition
        self.perfect_hashing = perfect_hashing

    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        msg = proto.physical_plan.PhysicalRelation()
        a = msg.concatenate_aggregate
        a.child.CopyFrom(ctx.emit(self.input))
        for key in self.keys:
            a.keys.add().CopyFrom(key.to_proto())
        for kind_str, expr in self.measures:
            val = a.values.add()
            val.aggregation_kind = _AGG_KIND_MAP[kind_str]
            val.expression.CopyFrom(expr.to_proto())
        if self.condition is not None:
            a.condition.CopyFrom(self.condition.to_proto())
        a.perfect_hashing = self.perfect_hashing
        return msg


class SortRelation(Relation):
    """Sort by ``(expression, order, null_precedence)`` keys.

    ``__init__`` raises ValueError if ``order`` or ``null_precedence``
    isn't a known value.
    """

    def __init__(
        self,
        input: Relation,
        keys: list[tuple[Expression, str, str]],
    ) -> None:
        for _, order, null_order in keys:
            if order not in _ORDER_MAP:
                raise ValueError(f"Unknown sort order: {order}")
            if null_order not in _NULL_ORDER_MAP:
                raise ValueError(f"Unknown null precedence: {null_order}")
        self.input = input
        self.keys = keys

    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        msg = proto.physical_plan.PhysicalRelation()
        s = msg.concatenate_sort
        s.child.CopyFrom(ctx.emit(self.input))
        for expr, order, null_order in self.keys:
            s.keys.add().CopyFrom(expr.to_proto())
            s.column_orders.append(_ORDER_MAP[order])
            s.null_precedences.append(_NULL_ORDER_MAP[null_order])
        return msg


class FetchRelation(Relation):
    """Limit/offset window over an input relation."""

    def __init__(self, input: Relation, offset: int, count: int) -> None:
        self.input = input
        self.offset = offset
        self.count = count

    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        msg = proto.physical_plan.PhysicalRelation()
        msg.fetch.child.CopyFrom(ctx.emit(self.input))
        msg.fetch.offset = self.offset
        msg.fetch.count = self.count
        return msg


class UnionAllRelation(Relation):
    """Concatenation of two inputs with the same schema."""

    def __init__(self, lhs: Relation, rhs: Relation) -> None:
        self.lhs = lhs
        self.rhs = rhs

    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        msg = proto.physical_plan.PhysicalRelation()
        msg.union_all.left.CopyFrom(ctx.emit(self.lhs))
        msg.union_all.right.CopyFrom(ctx.emit(self.rhs))
        return msg


class ShuffleRelation(Relation):
    """Shuffle (repartition) by the given expressions."""

    def __init__(self, input: Relation, shuffle_cols: list[Expression]) -> None:
        self.input = input
        self.shuffle_cols = shuffle_cols

    def to_proto(self, ctx: SerializationContext) -> proto.physical_plan.PhysicalRelation:
        msg = proto.physical_plan.PhysicalRelation()
        s = msg.shuffle
        s.child.CopyFrom(ctx.emit(self.input))
        for expr in self.shuffle_cols:
            s.shuffle_cols.add().CopyFrom(expr.to_proto())
        return msg
