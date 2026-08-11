#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

"""
Tests for the physical_plan package: protobuf roundtrip serialization,
enum alignment, error paths, and query registry.

The protos come from the gqe checkout via the install, so these need a
GQE_FETCH_ENGINE=ON build.

Usage:
    pytest gqe_bench/tests/integration/test_physical_plan.py -v
"""

import struct

import numpy as np
import pytest

from gqe_bench.physical_plan import proto
from gqe_bench.physical_plan.expression import (
    _COMPONENT_MAP,
    Cast,
    ColumnReference,
    DataType,
    DataTypeId,
    DateLiteral,
    DatePartExpr,
    DateTimeComponent,
    DecimalLiteral,
    Expression,
    FixedPointLiteral,
    IfThenElseExpr,
    LikeExpr,
    Literal,
    SubstrExpr,
)
from gqe_bench.physical_plan.relation import (
    _AGG_KIND_MAP,
    _JOIN_TYPE_MAP,
    _NULL_ORDER_MAP,
    _ORDER_MAP,
    AggregationKind,
    JoinType,
    NullOrder,
    Order,
    ReadRelation,
    SerializationContext,
    UniqueKeysPolicy,
)

_pp_pb2 = proto.physical_plan
_expr_pb2 = proto.expression
_result_pb2 = proto.result


class TestEnumAlignment:
    """Verify Python enum values match proto definitions."""

    def test_data_type_id_matches_proto(self) -> None:
        for member in DataTypeId:
            proto_val = getattr(_result_pb2, member.name, None)
            assert proto_val is not None, f"DataTypeId.{member.name} not in proto"
            assert (
                member.value == proto_val
            ), f"DataTypeId.{member.name}={member.value} != proto {proto_val}"

    def test_component_map_covers_enum(self) -> None:
        for member in DateTimeComponent:
            assert (
                member.name.lower() in _COMPONENT_MAP
            ), f"DateTimeComponent.{member.name} missing from _COMPONENT_MAP"

    def test_join_type_map_values_are_valid(self) -> None:
        for name, value in _JOIN_TYPE_MAP.items():
            assert isinstance(value, JoinType), f"_JOIN_TYPE_MAP['{name}'] is not JoinType"

    def test_agg_kind_map_values_are_valid(self) -> None:
        for name, value in _AGG_KIND_MAP.items():
            assert isinstance(
                value, AggregationKind
            ), f"_AGG_KIND_MAP['{name}'] is not AggregationKind"

    def test_order_map_values_are_valid(self) -> None:
        for name, value in _ORDER_MAP.items():
            assert isinstance(value, Order)
        for name, value in _NULL_ORDER_MAP.items():
            assert isinstance(value, NullOrder)


class TestExpressionRoundtrip:
    def test_column_reference(self) -> None:
        cr = ColumnReference(5)
        msg = cr.to_proto()
        assert msg.column_reference.column_idx == 5
        parsed = _expr_pb2.Expression.FromString(cr.serialize())
        assert parsed == msg

    def test_literal_int32(self) -> None:
        lit = Literal(42)
        msg = lit.to_proto()
        assert msg.literal.data_type.type_id == DataTypeId.INT32
        assert msg.literal.value == struct.pack("<i", 42)

    def test_literal_int64_auto_promotion(self) -> None:
        lit = Literal(5_000_000_000)
        msg = lit.to_proto()
        assert msg.literal.data_type.type_id == DataTypeId.INT64
        assert msg.literal.value == struct.pack("<q", 5_000_000_000)

    def test_literal_np_int64(self) -> None:
        lit = Literal(np.int64(99))
        msg = lit.to_proto()
        assert msg.literal.data_type.type_id == DataTypeId.INT64
        assert msg.literal.value == struct.pack("<q", 99)

    def test_literal_np_float32(self) -> None:
        lit = Literal(np.float32(1.5))
        msg = lit.to_proto()
        assert msg.literal.data_type.type_id == DataTypeId.FLOAT32
        assert msg.literal.value == struct.pack("<f", 1.5)

    def test_literal_string(self) -> None:
        lit = Literal("hello")
        msg = lit.to_proto()
        assert msg.literal.data_type.type_id == DataTypeId.STRING
        assert msg.literal.value == b"hello"

    def test_literal_float64(self) -> None:
        lit = Literal(3.14)
        msg = lit.to_proto()
        assert msg.literal.data_type.type_id == DataTypeId.FLOAT64
        assert msg.literal.value == struct.pack("<d", 3.14)

    def test_date_literal(self) -> None:
        dl = DateLiteral("1998-09-02")
        msg = dl.to_proto()
        assert msg.literal.data_type.type_id == DataTypeId.TIMESTAMP_DAYS
        parsed = _expr_pb2.Expression.FromString(dl.serialize())
        assert parsed == msg

    def test_binary_op_field_values(self) -> None:
        expr = ColumnReference(0) + ColumnReference(1)
        msg = expr.to_proto()
        assert msg.binary_op.left.column_reference.column_idx == 0
        assert msg.binary_op.right.column_reference.column_idx == 1

    def test_comparison_operators_roundtrip(self) -> None:
        cr = ColumnReference(0)
        dl = DateLiteral("1995-01-01")
        for op in [cr == dl, cr != dl, cr < dl, cr <= dl, cr > dl, cr >= dl]:
            assert isinstance(op, Expression)
            parsed = _expr_pb2.Expression.FromString(op.serialize())
            assert parsed == op.to_proto()

    def test_logical_operators(self) -> None:
        a = ColumnReference(0) == Literal(1)
        b = ColumnReference(1) == Literal(2)
        for expr in [a & b, a | b]:
            parsed = _expr_pb2.Expression.FromString(expr.serialize())
            assert parsed == expr.to_proto()

    def test_like_expr(self) -> None:
        expr = LikeExpr(ColumnReference(0), "%green%")
        msg = expr.to_proto()
        assert msg.scalar_function.pattern == "%green%"
        parsed = _expr_pb2.Expression.FromString(expr.serialize())
        assert parsed == msg

    def test_substr_expr(self) -> None:
        expr = SubstrExpr(ColumnReference(0), 1, 2)
        msg = expr.to_proto()
        assert msg.scalar_function.start == 1
        assert msg.scalar_function.length == 2

    def test_cast(self) -> None:
        expr = Cast(ColumnReference(0), DataType(DataTypeId.FLOAT64))
        msg = expr.to_proto()
        assert msg.cast.target_type.type_id == DataTypeId.FLOAT64
        parsed = _expr_pb2.Expression.FromString(expr.serialize())
        assert parsed == msg

    def test_date_part(self) -> None:
        expr = DatePartExpr(ColumnReference(0), "year")
        msg = expr.to_proto()
        assert msg.scalar_function.datetime_component == DateTimeComponent.YEAR
        parsed = _expr_pb2.Expression.FromString(expr.serialize())
        assert parsed == msg

    def test_if_then_else_field_values(self) -> None:
        expr = IfThenElseExpr(
            ColumnReference(0) == Literal(1),
            Literal(10),
            Literal(20),
        )
        msg = expr.to_proto()
        assert msg.if_then_else.then_expr.literal.value == struct.pack("<i", 10)
        assert msg.if_then_else.else_expr.literal.value == struct.pack("<i", 20)
        parsed = _expr_pb2.Expression.FromString(expr.serialize())
        assert parsed == msg


class TestRelationRoundtrip:
    def test_read_relation(self) -> None:
        r = ReadRelation("lineitem", ["l_orderkey"], [DataType(DataTypeId.INT64)])
        msg = SerializationContext().emit(r)
        assert msg.read.table_name == "lineitem"
        assert list(msg.read.column_names) == ["l_orderkey"]
        assert msg.read.data_types[0].type_id == DataTypeId.INT64
        parsed = _pp_pb2.PhysicalRelation.FromString(r.serialize())
        assert parsed == msg

    def test_read_with_partial_filter(self) -> None:
        r = ReadRelation(
            "lineitem",
            ["l_shipdate"],
            [DataType(DataTypeId.TIMESTAMP_DAYS)],
            partial_filter=ColumnReference(10) <= DateLiteral("1998-09-02"),
        )
        msg = SerializationContext().emit(r)
        assert msg.read.HasField("partial_filter")
        parsed = _pp_pb2.PhysicalRelation.FromString(r.serialize())
        assert parsed == msg

    def test_filter_relation(self) -> None:
        r = ReadRelation("t", ["a", "b"], [DataType(DataTypeId.INT32), DataType(DataTypeId.INT32)])
        f = r.filter(ColumnReference(0) == Literal(1), [0, 1])
        msg = SerializationContext().emit(f)
        assert list(msg.filter.projection_indices) == [0, 1]
        parsed = _pp_pb2.PhysicalRelation.FromString(f.serialize())
        assert parsed == msg

    def test_broadcast_join(self) -> None:
        left = ReadRelation("a", ["x"], [DataType(DataTypeId.INT32)])
        right = ReadRelation("b", ["y"], [DataType(DataTypeId.INT32)])
        j = left.broadcast_join(
            right,
            ColumnReference(0) == ColumnReference(1),
            [0, 1],
            "inner",
            unique_keys_policy=UniqueKeysPolicy.RIGHT,
        )
        msg = SerializationContext().emit(j)
        assert msg.broadcast_join.join_type == JoinType.INNER
        assert msg.broadcast_join.unique_keys_policy == UniqueKeysPolicy.RIGHT
        assert list(msg.broadcast_join.projection_indices) == [0, 1]
        assert msg.broadcast_join.left.read.table_name == "a"
        assert msg.broadcast_join.right.read.table_name == "b"
        parsed = _pp_pb2.PhysicalRelation.FromString(j.serialize())
        assert parsed == msg

    def test_shuffle_join(self) -> None:
        left = ReadRelation("a", ["x"], [DataType(DataTypeId.INT32)])
        right = ReadRelation("b", ["y"], [DataType(DataTypeId.INT32)])
        j = left.shuffle_join(
            right,
            ColumnReference(0) == ColumnReference(1),
            [0, 1],
            "left_semi",
        )
        msg = SerializationContext().emit(j)
        assert msg.shuffle_join.join_type == JoinType.LEFT_SEMI
        parsed = _pp_pb2.PhysicalRelation.FromString(j.serialize())
        assert parsed == msg

    def test_aggregate_relation(self) -> None:
        r = ReadRelation(
            "t", ["a", "b"], [DataType(DataTypeId.INT32), DataType(DataTypeId.FLOAT64)]
        )
        agg = r.aggregate([ColumnReference(0)], [("sum", ColumnReference(1))])
        msg = SerializationContext().emit(agg)
        assert msg.concatenate_aggregate.values[0].aggregation_kind == AggregationKind.SUM
        parsed = _pp_pb2.PhysicalRelation.FromString(agg.serialize())
        assert parsed == msg

    def test_sort_relation(self) -> None:
        r = ReadRelation("t", ["a"], [DataType(DataTypeId.INT32)])
        s = r.sort([(ColumnReference(0), "descending", "before")])
        msg = SerializationContext().emit(s)
        assert list(msg.concatenate_sort.column_orders) == [Order.DESCENDING]
        assert list(msg.concatenate_sort.null_precedences) == [NullOrder.BEFORE]
        parsed = _pp_pb2.PhysicalRelation.FromString(s.serialize())
        assert parsed == msg

    def test_fetch_relation(self) -> None:
        r = ReadRelation("t", ["a"], [DataType(DataTypeId.INT32)])
        f = r.sort([(ColumnReference(0), "ascending", "before")]).fetch(0, 10)
        msg = SerializationContext().emit(f)
        assert msg.fetch.offset == 0
        assert msg.fetch.count == 10

    def test_project_relation(self) -> None:
        r = ReadRelation(
            "t", ["a", "b"], [DataType(DataTypeId.INT32), DataType(DataTypeId.FLOAT64)]
        )
        p = r.project([ColumnReference(0), ColumnReference(1) * Literal(2.0)])
        msg = SerializationContext().emit(p)
        assert len(msg.project.output_expressions) == 2
        parsed = _pp_pb2.PhysicalRelation.FromString(p.serialize())
        assert parsed == msg

    def test_union_all(self) -> None:
        a = ReadRelation("a", ["x"], [DataType(DataTypeId.INT32)])
        b = ReadRelation("b", ["y"], [DataType(DataTypeId.INT32)])
        u = a.union_all(b)
        msg = SerializationContext().emit(u)
        assert msg.union_all.left.read.table_name == "a"
        assert msg.union_all.right.read.table_name == "b"
        parsed = _pp_pb2.PhysicalRelation.FromString(u.serialize())
        assert parsed == msg

    def test_shuffle_relation(self) -> None:
        r = ReadRelation("t", ["a", "b"], [DataType(DataTypeId.INT32), DataType(DataTypeId.INT32)])
        s = r.shuffle([ColumnReference(0)])
        msg = SerializationContext().emit(s)
        assert len(msg.shuffle.shuffle_cols) == 1
        parsed = _pp_pb2.PhysicalRelation.FromString(s.serialize())
        assert parsed == msg

    def test_method_chaining(self) -> None:
        r = ReadRelation(
            "t", ["a", "b"], [DataType(DataTypeId.INT32), DataType(DataTypeId.FLOAT64)]
        )
        plan = (
            r.filter(ColumnReference(0) > Literal(0), [0, 1])
            .aggregate([ColumnReference(0)], [("sum", ColumnReference(1))])
            .sort([(ColumnReference(0), "ascending", "after")])
        )
        parsed = _pp_pb2.PhysicalRelation.FromString(plan.serialize())
        assert parsed == SerializationContext().emit(plan)


class TestErrorPaths:
    def test_invalid_join_type_raises(self) -> None:
        left = ReadRelation("a", ["x"], [DataType(DataTypeId.INT32)])
        right = ReadRelation("b", ["y"], [DataType(DataTypeId.INT32)])
        with pytest.raises(ValueError, match="Unknown join type"):
            left.broadcast_join(right, ColumnReference(0) == ColumnReference(1), [0], "bogus")

    def test_invalid_agg_kind_raises(self) -> None:
        r = ReadRelation("t", ["a"], [DataType(DataTypeId.INT32)])
        with pytest.raises(ValueError, match="Unknown aggregation kind"):
            r.aggregate([], [("bogus", ColumnReference(0))])

    def test_invalid_sort_order_raises(self) -> None:
        r = ReadRelation("t", ["a"], [DataType(DataTypeId.INT32)])
        with pytest.raises(ValueError, match="Unknown sort order"):
            r.sort([(ColumnReference(0), "bogus", "before")])

    def test_invalid_null_order_raises(self) -> None:
        r = ReadRelation("t", ["a"], [DataType(DataTypeId.INT32)])
        with pytest.raises(ValueError, match="Unknown null precedence"):
            r.sort([(ColumnReference(0), "ascending", "bogus")])

    def test_unsupported_literal_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported literal type"):
            Literal(object()).to_proto()

    def test_invalid_date_part_component_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown date-time component"):
            DatePartExpr(ColumnReference(0), "bogus")


class TestQueryRegistry:
    def test_lookup_base_query(self) -> None:
        from gqe_bench.suites.tpch.queries import lookup

        assert callable(lookup("1"))

    def test_lookup_fused_filter_variant(self) -> None:
        from gqe_bench.suites.tpch.queries import lookup

        assert callable(lookup("3_fused_filter"))

    def test_lookup_nonexistent_raises(self) -> None:
        from gqe_bench.suites.tpch.queries import lookup

        with pytest.raises(ValueError, match="No handcoded query module"):
            lookup("999")

    def test_all_base_queries_have_valid_structure(self) -> None:
        """Base queries produce parseable plans with expected root relation types."""
        from gqe_bench.suites.tpch.queries import lookup
        from gqe_bench.suites.tpch.table_schema import TpchTableSchema

        schema = TpchTableSchema("int64")
        # Skip Q8 and Q14 — no implementation in the suite.
        for q in [str(i) for i in range(1, 23) if i not in (8, 14)]:
            plan = lookup(q)(schema)
            msg = SerializationContext().emit(plan)
            parsed = _pp_pb2.PhysicalRelation.FromString(plan.serialize())
            assert parsed == msg, f"Q{q} roundtrip mismatch"
            # Verify it's not an empty/default message
            assert msg.ByteSize() > 0, f"Q{q} produced empty proto"


class TestTpchSuiteResolveContent:
    def test_handcoded_content_is_valid_plan(self) -> None:
        from gqe_bench.query_source import QuerySource
        from gqe_bench.suites.tpch import TpchSuite

        content = TpchSuite.resolve_content(
            "1",
            QuerySource.HANDCODED,
            None,
            1.0,
            "int64",
            load_all_data=True,
            decimal_type="double",
        )
        msg = _pp_pb2.PhysicalRelation.FromString(content)
        assert msg.ByteSize() > 0


class TestNarrowReadPartialFilters:
    def _collect_idxs(self, expr: object, out: list[int]) -> None:
        from gqe_bench.physical_plan.expression import ColumnReference

        if isinstance(expr, ColumnReference):
            out.append(expr.idx)
            return
        for attr in ("lhs", "rhs", "input", "if_expr", "then_expr", "else_expr"):
            child = getattr(expr, attr, None)
            if child is not None:
                self._collect_idxs(child, out)

    def _find_read(self, rel: object) -> object:
        from gqe_bench.physical_plan.relation import ReadRelation

        if isinstance(rel, ReadRelation):
            return rel
        for attr in ("input", "left_table", "right_table", "lhs", "rhs"):
            child = getattr(rel, attr, None)
            if child is not None:
                found = self._find_read(child)
                if found is not None:
                    return found
        return None

    def test_q6_read_filter_indices_remapped_to_loaded_positions(self) -> None:
        from gqe_bench.physical_plan.projection import narrow_read_partial_filters
        from gqe_bench.suites.tpch.queries import lookup
        from gqe_bench.suites.tpch.table_schema import TpchTableSchema

        schema = TpchTableSchema("int64")
        root = lookup("6")(schema, 1.0)
        read = self._find_read(root)

        # q6 authors the read filter against full lineitem positions:
        # l_quantity=4, l_discount=6, l_shipdate=10.
        before: list[int] = []
        self._collect_idxs(read.partial_filter, before)
        assert sorted(before) == [4, 6, 6, 10, 10]

        narrow_read_partial_filters(root, schema.column_orders())

        # Loaded order (full order filtered to the read's columns):
        # l_quantity=0, l_extendedprice=1, l_discount=2, l_shipdate=3.
        after: list[int] = []
        self._collect_idxs(read.partial_filter, after)
        assert sorted(after) == [0, 2, 2, 3, 3]

    def test_every_handcoded_plan_narrows_without_error(self) -> None:
        """Every handcoded plan's partial filters use only expression nodes the
        remap handles — narrowing must not raise for any of them."""
        from gqe_bench.physical_plan.projection import narrow_read_partial_filters
        from gqe_bench.query_source import QuerySource
        from gqe_bench.suites.tpch import TpchSuite
        from gqe_bench.suites.tpch.queries import lookup
        from gqe_bench.suites.tpch.table_schema import TpchTableSchema

        schema = TpchTableSchema("int64")
        full = schema.column_orders()
        for name in TpchSuite.available_queries(QuerySource.HANDCODED, None):
            narrow_read_partial_filters(lookup(name)(schema, 1.0), full)


class TestFixedPointEncoding:
    """The engine memcpys the literal's bytes straight into numeric::decimalN::rep,
    so a wrong width or a lost sign is silent corruption rather than an error."""

    def test_negative_rep_sign_extends_to_decimal64(self) -> None:
        msg = DecimalLiteral("-0.01", DataType(DataTypeId.DECIMAL64, -2)).to_proto()
        assert msg.literal.value == b"\xff" * 8
        assert msg.literal.data_type.type_id == DataTypeId.DECIMAL64

    def test_negative_rep_sign_extends_across_the_64_bit_boundary(self) -> None:
        msg = DecimalLiteral("-0.01", DataType(DataTypeId.DECIMAL128, -2)).to_proto()
        assert msg.literal.value == b"\xff" * 16

    def test_positive_rep_is_little_endian(self) -> None:
        # 12.34 at scale -2 is rep 1234.
        msg = DecimalLiteral("12.34", DataType(DataTypeId.DECIMAL64, -2)).to_proto()
        assert msg.literal.value == struct.pack("<q", 1234)

    def test_width_follows_the_declared_type_not_the_magnitude(self) -> None:
        small = DecimalLiteral("1", DataType(DataTypeId.DECIMAL128, -2)).to_proto()
        assert len(small.literal.value) == 16

    def test_rep_too_large_for_decimal128_raises(self) -> None:
        # Reached directly: _decimal_to_scaled_int rounds to the decimal context's
        # 28 significant digits, so a value this size arrives at _pick_type_id
        # already truncated below the 128-bit limit.
        with pytest.raises(OverflowError):
            FixedPointLiteral._pick_type_id(2**127)

    def test_width_selection_boundaries(self) -> None:
        assert FixedPointLiteral._pick_type_id(2**31 - 1) == DataTypeId.DECIMAL32
        assert FixedPointLiteral._pick_type_id(2**31) == DataTypeId.DECIMAL64
        assert FixedPointLiteral._pick_type_id(2**63) == DataTypeId.DECIMAL128


class TestDecimalScaleOnTheWire:
    """`oneof precision` makes an unset scale distinguishable from an explicit
    zero; the engine relies on that to reject a scale-less fixed-point type."""

    def test_decimal_literal_sets_scale(self) -> None:
        msg = DecimalLiteral("1", DataType(DataTypeId.DECIMAL64, -2)).to_proto()
        assert msg.literal.data_type.HasField("scale")
        assert msg.literal.data_type.scale == -2

    def test_non_decimal_literal_leaves_scale_unset(self) -> None:
        msg = Literal(1.0).to_proto()
        assert not msg.literal.data_type.HasField("scale")

    def test_cast_to_decimal_sets_scale(self) -> None:
        msg = Cast(ColumnReference(0), DataType(DataTypeId.DECIMAL64, -2)).to_proto()
        assert msg.cast.target_type.HasField("scale")
        assert msg.cast.target_type.scale == -2

    def test_cast_to_non_decimal_leaves_scale_unset(self) -> None:
        msg = Cast(ColumnReference(0), DataType(DataTypeId.FLOAT64)).to_proto()
        assert not msg.cast.target_type.HasField("scale")

    def test_read_relation_emits_scale_per_column(self) -> None:
        read = ReadRelation(
            "lineitem",
            ["l_orderkey", "l_quantity"],
            [DataType(DataTypeId.INT64), DataType(DataTypeId.DECIMAL64, -2)],
        )
        msg = read.to_proto(SerializationContext())
        assert not msg.read.data_types[0].HasField("scale")
        assert msg.read.data_types[1].HasField("scale")
        assert msg.read.data_types[1].scale == -2

    def test_decimal_type_without_scale_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            DataType(DataTypeId.DECIMAL64)

    def test_scale_on_a_non_decimal_type_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            DataType(DataTypeId.FLOAT64, -2)


class TestNarrowingAcceptsDecimalLiterals:
    def test_remap_walks_past_a_fixed_point_literal(self) -> None:
        from gqe_bench.physical_plan.projection import narrow_read_partial_filters
        from gqe_bench.suites.tpch.table_schema import TpchTableSchema

        # Partial-filter indices are authored against the full base-table order.
        full = TpchTableSchema("int64").column_orders()
        dec = DataType(DataTypeId.DECIMAL64, -2)
        read = ReadRelation(
            "lineitem",
            ["l_quantity"],
            [dec],
            ColumnReference(full["lineitem"].index("l_quantity")) > DecimalLiteral("1", dec),
        )
        narrow_read_partial_filters(read, full)
        # l_quantity is the only loaded column, so its index moves to 0 and the
        # decimal literal on the other side is walked past rather than raising.
        assert read.partial_filter.lhs.idx == 0
        assert isinstance(read.partial_filter.rhs, DecimalLiteral)


class TestDecimalTypeSelectsPlanShape:
    def test_handcoded_payload_differs_per_representation(self) -> None:
        from gqe_bench.query_source import QuerySource
        from gqe_bench.suites.tpch import TpchSuite

        def content(decimal_type: str) -> bytes:
            return TpchSuite.resolve_content(
                "1",
                QuerySource.HANDCODED,
                None,
                1.0,
                "int64",
                load_all_data=True,
                decimal_type=decimal_type,
            )

        assert content("double") != content("decimal")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
