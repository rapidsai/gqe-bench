# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retarget read partial-filter column indices for a column-projected load."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from gqe_bench.physical_plan.expression import (
    Cast,
    ColumnReference,
    DateLiteral,
    DatePartExpr,
    DecimalLiteral,
    Expression,
    FixedPointLiteral,
    IfThenElseExpr,
    LikeExpr,
    Literal,
    SubstrExpr,
    _BinaryExpr,
)
from gqe_bench.physical_plan.relation import (
    BroadcastJoinRelation,
    ReadRelation,
    Relation,
    ShuffleJoinRelation,
    UnionAllRelation,
)


def _walk_reads(relation: Relation) -> Iterator[ReadRelation]:
    """Yield every ``ReadRelation`` in ``relation``'s tree."""
    if isinstance(relation, ReadRelation):
        yield relation
    elif isinstance(relation, (BroadcastJoinRelation, ShuffleJoinRelation)):
        yield from _walk_reads(relation.left_table)
        yield from _walk_reads(relation.right_table)
    elif isinstance(relation, UnionAllRelation):
        yield from _walk_reads(relation.lhs)
        yield from _walk_reads(relation.rhs)
    else:
        yield from _walk_reads(relation.input)


def _remap(expression: Expression, full: list[str], loaded: list[str]) -> None:
    """Rewrite each ``ColumnReference`` from its ``full`` position to its ``loaded`` position.

    Raises TypeError on an unrecognized expression node rather than walking past
    it: a new node that can hold a ``ColumnReference`` must be handled here, or a
    stale index would surface only at query time instead of loudly at discovery.
    """
    if isinstance(expression, ColumnReference):
        expression.idx = loaded.index(full[expression.idx])
    elif isinstance(expression, _BinaryExpr):
        _remap(expression.lhs, full, loaded)
        _remap(expression.rhs, full, loaded)
    elif isinstance(expression, (Cast, LikeExpr, SubstrExpr, DatePartExpr)):
        _remap(expression.input, full, loaded)
    elif isinstance(expression, IfThenElseExpr):
        _remap(expression.if_expr, full, loaded)
        _remap(expression.then_expr, full, loaded)
        _remap(expression.else_expr, full, loaded)
    elif not isinstance(expression, (Literal, DateLiteral, FixedPointLiteral, DecimalLiteral)):
        raise TypeError(
            f"narrow_read_partial_filters: unhandled expression type "
            f"{type(expression).__name__}"
        )


def narrow_read_partial_filters(root: Relation, full_columns: dict[str, list[str]]) -> None:
    """Retarget read partial-filter column indices to a narrowed load, in place.

    Handcoded reads author partial-filter ``ColumnReference`` indices against the
    full base-table column order (``full_columns[table]``). When a table is loaded
    with only a subset of its columns — the union of that table's reads' ``columns``
    taken in full order — each such index must shift to the column's position in the
    loaded subset. Filter and aggregate conditions index the read's output, not the
    base table, and are left untouched.

    A read may appear in more than one place in a plan (reused subtrees); each is
    retargeted once.
    """
    loaded_names: dict[str, set[str]] = defaultdict(set)
    for read in _walk_reads(root):
        loaded_names[read.table].update(read.columns)
    loaded_columns = {
        table: [c for c in full_columns[table] if c in names]
        for table, names in loaded_names.items()
    }

    fixed: set[int] = set()
    for read in _walk_reads(root):
        if read.partial_filter is not None and id(read) not in fixed:
            fixed.add(id(read))
            _remap(read.partial_filter, full_columns[read.table], loaded_columns[read.table])
