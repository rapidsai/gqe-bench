# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expression tree for physical plan construction.

Serializes via grpc.protos()-generated message classes from
physical_plan.proto. Provides an operator-overloading API for building
expression trees.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import struct
from abc import ABC, abstractmethod
from enum import IntEnum

import numpy as np

from gqe_bench.physical_plan import proto


class BinaryOp(IntEnum):
    """Binary operator opcodes (matches the proto's BinaryOp enum)."""

    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    TRUE_DIV = 4
    FLOOR_DIV = 5  # noqa: E702
    MOD = 6
    PMOD = 7
    PYMOD = 8
    POW = 9
    INT_POW = 10
    LOG_BASE = 11  # noqa: E702
    ATAN2 = 12
    SHIFT_LEFT = 13
    SHIFT_RIGHT = 14
    SHIFT_RIGHT_UNSIGNED = 15  # noqa: E702
    BITWISE_AND = 16
    BITWISE_OR = 17
    BITWISE_XOR = 18  # noqa: E702
    LOGICAL_AND = 19
    LOGICAL_OR = 20  # noqa: E702
    EQUAL = 21
    NOT_EQUAL = 22
    LESS = 23
    GREATER = 24  # noqa: E702
    LESS_EQUAL = 25
    GREATER_EQUAL = 26  # noqa: E702
    NULL_EQUALS = 27
    NULL_NOT_EQUALS = 28
    NULL_MAX = 29
    NULL_MIN = 30  # noqa: E702
    GENERIC_BINARY = 31
    NULL_LOGICAL_AND = 32
    NULL_LOGICAL_OR = 33  # noqa: E702


class ScalarFunctionKind(IntEnum):
    """Scalar functions exposed via the ScalarFunction proto message."""

    DATEPART = 0
    LIKE = 1
    SUBSTR = 2
    ROUND = 3  # noqa: E702


class DateTimeComponent(IntEnum):
    """Date / time fields extractable via the DATEPART scalar function."""

    YEAR = 0
    MONTH = 1
    DAY = 2
    WEEKDAY = 3
    HOUR = 4
    MINUTE = 5  # noqa: E702
    SECOND = 6
    MILLISECOND = 7
    MICROSECOND = 8
    NANOSECOND = 9  # noqa: E702


class DataTypeId(IntEnum):
    """Column data-type identifiers (matches the proto's DataType enum, which mirrors cudf's type_id)."""

    EMPTY = 0
    INT8 = 1
    INT16 = 2
    INT32 = 3
    INT64 = 4  # noqa: E702
    UINT8 = 5
    UINT16 = 6
    UINT32 = 7
    UINT64 = 8  # noqa: E702
    FLOAT32 = 9
    FLOAT64 = 10
    BOOL8 = 11  # noqa: E702
    TIMESTAMP_DAYS = 12
    TIMESTAMP_SECONDS = 13  # noqa: E702
    TIMESTAMP_MILLISECONDS = 14
    TIMESTAMP_MICROSECONDS = 15  # noqa: E702
    TIMESTAMP_NANOSECONDS = 16  # noqa: E702
    DURATION_DAYS = 17
    DURATION_SECONDS = 18  # noqa: E702
    DURATION_MILLISECONDS = 19
    DURATION_MICROSECONDS = 20  # noqa: E702
    DURATION_NANOSECONDS = 21  # noqa: E702
    DICTIONARY32 = 22
    STRING = 23
    LIST = 24  # noqa: E702
    DECIMAL32 = 25
    DECIMAL64 = 26
    DECIMAL128 = 27
    STRUCT = 28  # noqa: E702


_DECIMAL_TYPE_IDS: frozenset[DataTypeId] = frozenset(
    {DataTypeId.DECIMAL32, DataTypeId.DECIMAL64, DataTypeId.DECIMAL128}
)


@dataclasses.dataclass(frozen=True)
class DataType:
    """A column type: a ``DataTypeId`` plus a scale for the decimal types.

    Mirrors the proto's ``DataType{type_id, oneof precision {scale}}``. The
    ``oneof`` makes an unset scale distinguishable from an explicit zero, so
    ``scale`` is emitted only for the decimal types and is required for them —
    a fixed-point type without one reads as scale 0 on the wire.
    """

    type_id: DataTypeId
    scale: int | None = None

    def __post_init__(self) -> None:
        """Raise ValueError if ``scale`` and ``type_id`` disagree about being decimal."""
        if self.type_id in _DECIMAL_TYPE_IDS:
            if self.scale is None:
                raise ValueError(f"{self.type_id.name} requires a scale")
        elif self.scale is not None:
            raise ValueError(f"scale is not meaningful for {self.type_id.name}")

    def to_proto(self) -> proto.data_type.DataType:
        """Build a gqe.proto.DataType message."""
        msg = proto.data_type.DataType()
        msg.type_id = self.type_id
        if self.scale is not None:
            msg.scale = self.scale
        return msg


class Expression(ABC):
    """Abstract base for physical-plan expression nodes.

    Subclasses implement ``to_proto`` to emit the matching ``Expression``
    oneof. Operator overloads on this base (``__eq__``, ``__add__``,
    ...) compose subclasses into a tree.
    """

    __hash__ = None  # __eq__ returns Expression, not bool

    @abstractmethod
    def to_proto(self) -> proto.expression.Expression:
        """Build a gqe.proto.Expression message."""

    def serialize(self) -> bytes:
        """Serialize as protobuf bytes."""
        return self.to_proto().SerializeToString()

    def __eq__(self, other: object) -> Expression:  # type: ignore[override]
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.EQUAL, self, other)
        return NotImplemented

    def __ne__(self, other: object) -> Expression:  # type: ignore[override]
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.NOT_EQUAL, self, other)
        return NotImplemented

    def __lt__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.LESS, self, other)
        return NotImplemented

    def __le__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.LESS_EQUAL, self, other)
        return NotImplemented

    def __gt__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.GREATER, self, other)
        return NotImplemented

    def __ge__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.GREATER_EQUAL, self, other)
        return NotImplemented

    def __and__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.NULL_LOGICAL_AND, self, other)
        return NotImplemented

    def __or__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.NULL_LOGICAL_OR, self, other)
        return NotImplemented

    def __mul__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.MUL, self, other)
        return NotImplemented

    def __truediv__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.TRUE_DIV, self, other)
        return NotImplemented

    def __add__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.ADD, self, other)
        return NotImplemented

    def __sub__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return _BinaryExpr(BinaryOp.SUB, self, other)
        return NotImplemented


class ColumnReference(Expression):
    """Reference to an input column by index."""

    def __init__(self, idx: int) -> None:
        self.idx = idx

    def to_proto(self) -> proto.expression.Expression:
        msg = proto.expression.Expression()
        msg.column_reference.column_idx = self.idx
        return msg


class _BinaryExpr(Expression):
    """Binary operator expression (BinaryOpExpr in proto)."""

    def __init__(self, op: BinaryOp, lhs: Expression, rhs: Expression) -> None:
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def to_proto(self) -> proto.expression.Expression:
        msg = proto.expression.Expression()
        msg.binary_op.op = self.op
        msg.binary_op.left.CopyFrom(self.lhs.to_proto())
        msg.binary_op.right.CopyFrom(self.rhs.to_proto())
        return msg


class AndExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.NULL_LOGICAL_AND, lhs, rhs)


class OrExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.NULL_LOGICAL_OR, lhs, rhs)


class EqualExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.EQUAL, lhs, rhs)


class NotEqualExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.NOT_EQUAL, lhs, rhs)


class LessExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.LESS, lhs, rhs)


class GreaterExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.GREATER, lhs, rhs)


class LessEqualExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.LESS_EQUAL, lhs, rhs)


class GreaterEqualExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.GREATER_EQUAL, lhs, rhs)


class MultiplyExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.MUL, lhs, rhs)


class DivideExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.TRUE_DIV, lhs, rhs)


class AddExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.ADD, lhs, rhs)


class SubtractExpr(_BinaryExpr):
    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        super().__init__(BinaryOp.SUB, lhs, rhs)


_LiteralValue = int | np.int32 | np.int64 | str | float | np.float32 | np.float64


class Literal(Expression):
    """Scalar literal value (int / float / str / numpy variants)."""

    def __init__(self, value: _LiteralValue) -> None:
        self.value = value

    def to_proto(self) -> proto.expression.Expression:
        type_id, value_bytes = _encode_literal_value(self.value)
        msg = proto.expression.Expression()
        msg.literal.data_type.type_id = type_id
        msg.literal.value = value_bytes
        return msg


def _encode_literal_value(
    value: _LiteralValue,
) -> tuple[int, bytes]:
    """Return (proto DataType enum value, little-endian bytes) for a literal.

    Raises TypeError for unsupported literal types.
    """
    if isinstance(value, str):
        return proto.result.STRING, value.encode("utf-8")
    if isinstance(value, np.float32):
        return proto.result.FLOAT32, struct.pack("<f", value)
    if isinstance(value, float | np.float64):
        return proto.result.FLOAT64, struct.pack("<d", value)
    if isinstance(value, np.int64):
        return proto.result.INT64, struct.pack("<q", value)
    if isinstance(value, int | np.int32):
        # np.int32 is always 32-bit; plain int auto-promotes if out of range
        if isinstance(value, np.int32) or -2_147_483_648 <= value <= 2_147_483_647:
            return proto.result.INT32, struct.pack("<i", value)
        return proto.result.INT64, struct.pack("<q", value)
    raise TypeError(f"Unsupported literal type: {type(value)}")


_EPOCH = datetime.date(1970, 1, 1)


class DateLiteral(Expression):
    """ISO-format date literal stored as TIMESTAMP_DAYS since epoch."""

    def __init__(self, date_string: str) -> None:
        self.date_string = date_string

    def to_proto(self) -> proto.expression.Expression:
        days = (datetime.date.fromisoformat(self.date_string) - _EPOCH).days
        msg = proto.expression.Expression()
        msg.literal.data_type.type_id = proto.result.TIMESTAMP_DAYS
        msg.literal.value = struct.pack("<i", days)
        return msg


_DecimalValue = str | int | float | decimal.Decimal | np.integer | np.floating

# Rep limits used to choose decimal32/64/128.
_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_INT128_MIN, _INT128_MAX = -(2**127), 2**127 - 1

# struct format for each fixed-point rep width; decimal128 has no struct code.
_REP_FORMAT: dict[DataTypeId, str] = {
    DataTypeId.DECIMAL32: "<i",
    DataTypeId.DECIMAL64: "<q",
}


def _as_decimal(value: _DecimalValue) -> decimal.Decimal:
    """Coerce a decimal-like value to Decimal."""
    if isinstance(value, str):
        return decimal.Decimal(value)
    if isinstance(value, decimal.Decimal):
        return value
    if isinstance(value, (int, np.integer)):
        return decimal.Decimal(int(value))
    if isinstance(value, (float, np.floating)):
        # Convert through str to avoid binary float artifacts.
        return decimal.Decimal(str(float(value)))
    raise TypeError(f"Unsupported decimal literal value type: {type(value).__name__}")


def _decimal_to_scaled_int(value: _DecimalValue, scale: int) -> int:
    """Convert a decimal-like value to fixed-point rep for the given scale."""
    # Decimal.scaleb(n) returns d * 10**n.
    scaled = _as_decimal(value).scaleb(-scale)
    return int(scaled.to_integral_value(rounding=decimal.ROUND_HALF_EVEN))


def _infer_minimum_decimal_scale(value: _DecimalValue) -> int:
    """Infer the smallest non-positive scale that exactly represents ``value``."""
    if isinstance(value, (int, np.integer)):
        return 0

    exponent = _as_decimal(value).as_tuple().exponent
    # Keep scales non-positive for cuDF fixed-point.
    return int(exponent) if exponent <= 0 else 0


def _encode_fixed_point_rep(rep: int, type_id: DataTypeId) -> bytes:
    """Return the little-endian two's-complement bytes for a fixed-point rep.

    The engine ``memcpy``s these straight into ``numeric::decimalN::rep``, so the
    width must match the type and negative reps must sign-extend to it.
    """
    if type_id == DataTypeId.DECIMAL128:
        return rep.to_bytes(16, "little", signed=True)
    fmt = _REP_FORMAT.get(type_id)
    if fmt is None:
        raise ValueError(f"Unsupported fixed-point type_id: {type_id}")
    return struct.pack(fmt, rep)


class FixedPointLiteral(Expression):
    """Fixed-point literal for cuDF decimal32/64/128.

    ``type_id`` defaults to the narrowest width that holds the scaled rep.
    """

    def __init__(
        self,
        value: _DecimalValue,
        scale: int = -2,
        type_id: DataTypeId | None = None,
        is_null: bool = False,
    ) -> None:
        self.rep = _decimal_to_scaled_int(value, scale)
        self.scale = scale
        self.is_null = is_null
        self.type_id = type_id if type_id is not None else self._pick_type_id(self.rep)

    @staticmethod
    def _pick_type_id(rep: int) -> DataTypeId:
        """Return the narrowest decimal type whose rep holds ``rep``."""
        if _INT32_MIN <= rep <= _INT32_MAX:
            return DataTypeId.DECIMAL32
        if _INT64_MIN <= rep <= _INT64_MAX:
            return DataTypeId.DECIMAL64
        if _INT128_MIN <= rep <= _INT128_MAX:
            return DataTypeId.DECIMAL128
        raise OverflowError(f"Decimal rep {rep} does not fit in decimal128")

    def to_proto(self) -> proto.expression.Expression:
        msg = proto.expression.Expression()
        msg.literal.data_type.CopyFrom(DataType(self.type_id, self.scale).to_proto())
        msg.literal.value = _encode_fixed_point_rep(self.rep, self.type_id)
        msg.literal.is_null = self.is_null
        return msg


class DecimalLiteral(Expression):
    """Decimal-valued literal that lowers to float or fixed-point by target type.

    ``target`` carries the representation the surrounding plan uses. A
    ``DataType`` supplies the scale; a bare ``DataTypeId`` leaves it to be
    inferred from the value.
    """

    def __init__(self, value: _DecimalValue, target: DataTypeId | DataType) -> None:
        self.value = value
        if isinstance(target, DataType):
            self.type_id = target.type_id
            self.scale = target.scale
        else:
            self.type_id = target
            self.scale = None

    def to_proto(self) -> proto.expression.Expression:
        if self.type_id == DataTypeId.FLOAT64:
            return Literal(float(self.value)).to_proto()
        if self.type_id == DataTypeId.FLOAT32:
            return Literal(np.float32(self.value)).to_proto()
        if self.type_id in _DECIMAL_TYPE_IDS:
            scale = (
                self.scale if self.scale is not None else _infer_minimum_decimal_scale(self.value)
            )
            return FixedPointLiteral(self.value, scale=scale, type_id=self.type_id).to_proto()
        raise ValueError(
            f"DecimalLiteral does not support {self.type_id.name}; expected a float or decimal type"
        )


class Cast(Expression):
    """Cast an expression to ``target_type``."""

    def __init__(self, input: Expression, target_type: DataType) -> None:
        self.input = input
        self.target_type = target_type

    def to_proto(self) -> proto.expression.Expression:
        msg = proto.expression.Expression()
        msg.cast.input.CopyFrom(self.input.to_proto())
        msg.cast.target_type.CopyFrom(self.target_type.to_proto())
        return msg


class LikeExpr(Expression):
    """SQL LIKE with an optional escape character."""

    def __init__(self, input: Expression, pattern: str, escape_character: str = "") -> None:
        self.input = input
        self.pattern = pattern
        self.escape_character = escape_character

    def to_proto(self) -> proto.expression.Expression:
        msg = proto.expression.Expression()
        sf = msg.scalar_function
        sf.function_kind = ScalarFunctionKind.LIKE
        sf.pattern = self.pattern
        sf.escape_character = self.escape_character
        sf.arguments.add().CopyFrom(self.input.to_proto())
        return msg


class SubstrExpr(Expression):
    """SQL SUBSTR: extract a fixed-length window from a string expression."""

    def __init__(self, input: Expression, start: int, length: int) -> None:
        self.input = input
        self.start = start
        self.length = length

    def to_proto(self) -> proto.expression.Expression:
        msg = proto.expression.Expression()
        sf = msg.scalar_function
        sf.function_kind = ScalarFunctionKind.SUBSTR
        sf.start = self.start
        sf.length = self.length
        sf.arguments.add().CopyFrom(self.input.to_proto())
        return msg


_COMPONENT_MAP: dict[str, DateTimeComponent] = {
    "year": DateTimeComponent.YEAR,
    "month": DateTimeComponent.MONTH,
    "day": DateTimeComponent.DAY,
    "weekday": DateTimeComponent.WEEKDAY,
    "hour": DateTimeComponent.HOUR,
    "minute": DateTimeComponent.MINUTE,
    "second": DateTimeComponent.SECOND,
    "millisecond": DateTimeComponent.MILLISECOND,
    "microsecond": DateTimeComponent.MICROSECOND,
    "nanosecond": DateTimeComponent.NANOSECOND,
}


class DatePartExpr(Expression):
    """Extract a date/time component from a date expression.

    ``__init__`` raises ValueError if ``component`` is not one of
    the keys in ``_COMPONENT_MAP``.
    """

    def __init__(self, input: Expression, component: str) -> None:
        if component not in _COMPONENT_MAP:
            raise ValueError(f"Unknown date-time component: {component}")
        self.input = input
        self.component = component

    def to_proto(self) -> proto.expression.Expression:
        msg = proto.expression.Expression()
        sf = msg.scalar_function
        sf.function_kind = ScalarFunctionKind.DATEPART
        sf.datetime_component = _COMPONENT_MAP[self.component]
        sf.arguments.add().CopyFrom(self.input.to_proto())
        return msg


class IfThenElseExpr(Expression):
    """Conditional expression: evaluate ``then_expr`` when the condition is true, else ``else_expr``."""

    def __init__(self, if_expr: Expression, then_expr: Expression, else_expr: Expression) -> None:
        self.if_expr = if_expr
        self.then_expr = then_expr
        self.else_expr = else_expr

    def to_proto(self) -> proto.expression.Expression:
        msg = proto.expression.Expression()
        msg.if_then_else.condition.CopyFrom(self.if_expr.to_proto())
        msg.if_then_else.then_expr.CopyFrom(self.then_expr.to_proto())
        msg.if_then_else.else_expr.CopyFrom(self.else_expr.to_proto())
        return msg
