#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

"""
Tests for the proto-free parts of physical_plan.expression: decimal scaling and
scale inference.

Serialization tests live in tests/integration/test_physical_plan.py because they
need the installed protos; nothing here does.

Usage:
    pytest gqe_bench/tests/test_expression.py -v
"""

import pytest

from gqe_bench.physical_plan.expression import (
    _decimal_to_scaled_int,
    _infer_minimum_decimal_scale,
)


class TestDecimalRounding:
    """Scaling goes through Decimal, not float, and rounds half-to-even."""

    @pytest.mark.parametrize(
        "value,expected",
        [("0.005", 0), ("0.015", 2), ("0.025", 2), ("-0.005", 0), ("1.005", 100)],
    )
    def test_half_even_at_scale_minus_two(self, value: str, expected: int) -> None:
        assert _decimal_to_scaled_int(value, -2) == expected

    def test_float_input_routes_through_str(self) -> None:
        # 0.07 has no exact binary representation; going via str keeps rep at 7.
        assert _decimal_to_scaled_int(0.07, -2) == 7


class TestScaleInference:
    """A bare DataTypeId leaves the scale to be inferred from the value; q11
    relies on it so its 0.0001 / SF fraction is not rounded away."""

    @pytest.mark.parametrize(
        "value,expected",
        [("0.0001", -4), ("12.34", -2), ("1", 0), (7, 0), ("100", 0)],
    )
    def test_smallest_non_positive_scale(self, value: object, expected: int) -> None:
        assert _infer_minimum_decimal_scale(value) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
