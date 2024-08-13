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
In GQE, an **expression** is associated with a :mod:`relation <gqe.relation>` and evaluates to a
column. For example, a equality expression takes two expressions (columns) as input and returns
the element-wise comparison of the two input columns. It could be used as a join condition
associated with a :class:`join relation <gqe.relation.JoinRelation>`. An expression is represented
by an object of the :class:`Expression <gqe.expression.Expression>` class.
"""

import gqe.lib
import gqe.type
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Union  # Use | after migrating to Python>=3.10
import datetime


class Expression(ABC):
    @cached_property
    def _cpp(self) -> gqe.lib.Expression:
        """
        Convert `self` into the equivalent C++ object.

        Note that this method must be cached to allow multiple expressions share a common child
        expression.
        """
        return self._to_cpp()

    @abstractmethod
    def _to_cpp(self) -> gqe.lib.Expression:
        pass

    def __eq__(self, other):
        if isinstance(other, Expression):
            return EqualExpr(self, other)
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Expression):
            return NotEqualExpr(self, other)
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Expression):
            return LessExpr(self, other)
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, Expression):
            return LessEqualExpr(self, other)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Expression):
            return GreaterExpr(self, other)
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Expression):
            return GreaterEqualExpr(self, other)
        else:
            return NotImplemented

    def __and__(self, other):
        if isinstance(other, Expression):
            return AndExpr(self, other)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, Expression):
            return OrExpr(self, other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Expression):
            return MultiplyExpr(self, other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Expression):
            return DivideExpr(self, other)
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Expression):
            return AddExpr(self, other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Expression):
            return SubtractExpr(self, other)
        else:
            return NotImplemented


# TODO: It's more natural to use name references instead of position references.
class ColumnReference(Expression):
    def __init__(self, idx: int):
        """
        Construct a column reference expression.

        :param idx: Column index of the referenced column (zero-based).
        """
        self.idx = idx

    def _to_cpp(self):
        return gqe.lib.ColumnReference(self.idx)


class BinaryOpExpression(Expression):
    def __init__(self, lhs: Expression, rhs: Expression):
        """
        Construct a binary operator expression.

        :param lhs: Left-hand side of the expression.
        :param rhs: Right-hand side of the expression.
        """
        self.lhs = lhs
        self.rhs = rhs


class AndExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.And(self.lhs._cpp, self.rhs._cpp)


class OrExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.Or(self.lhs._cpp, self.rhs._cpp)


class EqualExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.Equal(self.lhs._cpp, self.rhs._cpp)


class NotEqualExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.NotEqual(self.lhs._cpp, self.rhs._cpp)


class LessExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.Less(self.lhs._cpp, self.rhs._cpp)


class GreaterExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.Greater(self.lhs._cpp, self.rhs._cpp)


class LessEqualExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.LessEqual(self.lhs._cpp, self.rhs._cpp)


class GreaterEqualExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.GreaterEqual(self.lhs._cpp, self.rhs._cpp)


class MultiplyExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.Multiply(self.lhs._cpp, self.rhs._cpp)


class DivideExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.Divide(self.lhs._cpp, self.rhs._cpp)


class AddExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.Add(self.lhs._cpp, self.rhs._cpp)


class SubtractExpr(BinaryOpExpression):
    def _to_cpp(self):
        return gqe.lib.Subtract(self.lhs._cpp, self.rhs._cpp)


class LikeExpr(Expression):
    def __init__(self, input: Expression, pattern: str, escape_character: str = ""):
        """
        Construct a like expression.

        :param input: Input strings to be filtered.
        :param patter: Like pattern to match within each string.
        :param escape_character: Optional character specifies the escape prefix. Could be an empty
            string for no escape character.
        """
        self.input = input
        self.pattern = pattern
        self.escape_character = escape_character

    def _to_cpp(self):
        return gqe.lib.Like(self.input._cpp, self.pattern, self.escape_character, False)


class IfThenElseExpr(Expression):
    def __init__(
        self, if_expr: Expression, then_expr: Expression, else_expr: Expression
    ):
        """
        Construct a if then else expression.

        :param if_expr: If expression
        :param then_expr: Then expression
        :pram else_expr: Else expression
        """
        self.if_expr = if_expr
        self.then_expr = then_expr
        self.else_expr = else_expr

    def _to_cpp(self):
        return gqe.lib.IfThenElse(
            self.if_expr._cpp, self.then_expr._cpp, self.else_expr._cpp
        )


class Literal(Expression):
    def __init__(self, value: Union[int, str, float]):
        """
        Construct a literal expression.

        :param value: Value of the literal. Currently, only integers, strings and floats are
            supported.
        """
        self.value = value

    def _to_cpp(self):
        if isinstance(self.value, str):
            return gqe.lib.LiteralString(self.value, False)
        elif isinstance(self.value, float):
            return gqe.lib.LiteralDouble(self.value, False)
        elif isinstance(self.value, int):
            return gqe.lib.LiteralInt64(self.value, False)
        else:
            raise NotImplementedError


class DateLiteral(Expression):
    def __init__(self, date_string: str):
        """
        Construct a date literal.

        :param date_string: Date in ISO 8601 format, e.g., '1995-03-15'.
        """
        self.date_string = date_string

    def _to_cpp(self):
        date = datetime.date.fromisoformat(self.date_string)
        return gqe.lib.date_from_days((date - datetime.date(1970, 1, 1)).days)


class Cast(Expression):
    def __init__(self, input: Expression, target_type: gqe.type.DataType):
        self.input = input
        self.target_type = target_type

    def _to_cpp(self):
        return gqe.lib.Cast(self.input._cpp, self.target_type._to_cpp())
