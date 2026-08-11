#!/usr/bin/env python3
#
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

"""
Tests for parquet validation and type coercion.

Usage:
    pytest gqe_bench/tests/test_validate.py -v
"""

import decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gqe_bench.validate import (
    _DECIMAL_ATOL,
    ValidationFailed,
    _assert_decimal_series_equal,
    _assert_frame_equal_decimal_aware,
    _convert_string_to_int,
    _is_decimal_series,
    _normalize_type,
    _to_decimal,
    validate_parquet,
)


def _write_parquet(path: Path, df: pd.DataFrame) -> Path:
    df.to_parquet(path)
    return path


class TestIsDecimalSeries:
    """A parquet DECIMAL column arrives as object dtype holding Decimal, so dtype
    alone cannot identify it."""

    def test_object_series_of_decimals(self) -> None:
        s = pd.Series([decimal.Decimal("1.00"), decimal.Decimal("2.00")])
        assert _is_decimal_series(s)

    def test_object_series_of_strings_is_not_decimal(self) -> None:
        assert not _is_decimal_series(pd.Series(["1.00", "2.00"]))

    def test_float_series_is_not_decimal(self) -> None:
        assert not _is_decimal_series(pd.Series([1.0, 2.0]))

    def test_all_null_series_is_not_decimal(self) -> None:
        # first_valid_index() is None, so there is no value to inspect.
        assert not _is_decimal_series(pd.Series([None, None], dtype=object))

    def test_leading_null_still_detected(self) -> None:
        s = pd.Series([None, decimal.Decimal("1.00")], dtype=object)
        assert _is_decimal_series(s)


class TestToDecimal:
    """Floats route through str so binary representation artifacts do not leak
    into the comparison."""

    def test_float_avoids_binary_artifact(self) -> None:
        assert _to_decimal(0.07) == decimal.Decimal("0.07")

    def test_decimal_passes_through(self) -> None:
        d = decimal.Decimal("1.23")
        assert _to_decimal(d) is d

    def test_int_and_numpy_int(self) -> None:
        assert _to_decimal(5) == decimal.Decimal(5)
        assert _to_decimal(np.int64(5)) == decimal.Decimal(5)

    def test_string_is_parsed(self) -> None:
        assert _to_decimal("1.25") == decimal.Decimal("1.25")


class TestAssertDecimalSeriesEqual:
    """Decimal columns are object dtype, so assert_series_equal cannot apply a
    tolerance to them; the comparison is done in Decimal arithmetic instead."""

    def _series(self, *values: str) -> pd.Series:
        return pd.Series([decimal.Decimal(v) for v in values])

    def test_equal_within_tolerance(self) -> None:
        _assert_decimal_series_equal(
            self._series("1.00", "2.00"), self._series("1.005", "2.00"), atol=_DECIMAL_ATOL
        )

    def test_difference_beyond_tolerance_raises(self) -> None:
        with pytest.raises(AssertionError, match="Decimal mismatch at index 0"):
            _assert_decimal_series_equal(
                self._series("1.00"), self._series("1.50"), atol=_DECIMAL_ATOL
            )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError, match="length mismatch"):
            _assert_decimal_series_equal(
                self._series("1.00"), self._series("1.00", "2.00"), atol=_DECIMAL_ATOL
            )

    def test_null_on_one_side_raises(self) -> None:
        lhs = pd.Series([decimal.Decimal("1.00"), None], dtype=object)
        rhs = self._series("1.00", "2.00")
        with pytest.raises(AssertionError, match="null mismatch"):
            _assert_decimal_series_equal(lhs, rhs, atol=_DECIMAL_ATOL)

    def test_nulls_on_both_sides_match(self) -> None:
        s = pd.Series([decimal.Decimal("1.00"), None], dtype=object)
        _assert_decimal_series_equal(s, s.copy(), atol=_DECIMAL_ATOL)


class TestAssertFrameEqualDecimalAware:
    """Decimal columns go through the Decimal path; everything else keeps pandas'
    own comparison."""

    def test_mixed_frame_compares_each_column_by_kind(self) -> None:
        lhs = pd.DataFrame({"d": [decimal.Decimal("1.00")], "f": [1.0], "s": ["a"]})
        rhs = pd.DataFrame({"d": [decimal.Decimal("1.005")], "f": [1.0], "s": ["a"]})
        _assert_frame_equal_decimal_aware(lhs, rhs, atol=_DECIMAL_ATOL)

    def test_non_decimal_column_difference_still_raises(self) -> None:
        lhs = pd.DataFrame({"d": [decimal.Decimal("1.00")], "f": [1.0]})
        rhs = pd.DataFrame({"d": [decimal.Decimal("1.00")], "f": [9.0]})
        with pytest.raises(AssertionError):
            _assert_frame_equal_decimal_aware(lhs, rhs, atol=_DECIMAL_ATOL)

    def test_column_mismatch_raises(self) -> None:
        lhs = pd.DataFrame({"a": [1]})
        rhs = pd.DataFrame({"b": [1]})
        with pytest.raises(AssertionError, match="Column mismatch"):
            _assert_frame_equal_decimal_aware(lhs, rhs, atol=_DECIMAL_ATOL)


class TestConvertStringToInt:
    """Tests for _convert_string_to_int helper."""

    def test_single_char(self) -> None:
        df = pd.DataFrame({"col": ["A", "B", "C"]})
        _convert_string_to_int(df, "col")
        assert df["col"].dtype == np.int8
        assert list(df["col"]) == [65, 66, 67]

    def test_multi_char_raises(self) -> None:
        df = pd.DataFrame({"col": ["AB", "CD"]})
        with pytest.raises(ValueError, match="single-char"):
            _convert_string_to_int(df, "col")


class TestNormalizeType:
    """Tests for _normalize_type helper."""

    def test_promotes_int32_to_int64(self) -> None:
        source = pd.DataFrame({"col": pd.array([1, 2], dtype=np.int64)})
        target = pd.DataFrame({"col": pd.array([1, 2], dtype=np.int32)})
        _normalize_type(source, target, "col")
        assert target["col"].dtype == np.int64

    def test_promotes_float32_to_float64(self) -> None:
        source = pd.DataFrame({"col": pd.array([1.0, 2.0], dtype=np.float64)})
        target = pd.DataFrame({"col": pd.array([1.0, 2.0], dtype=np.float32)})
        _normalize_type(source, target, "col")
        assert target["col"].dtype == np.float64


class TestValidateParquet:
    """Tests for validate_parquet end-to-end."""

    def test_identical_frames_pass(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        validate_parquet(
            _write_parquet(tmp_path / "test.parquet", df),
            _write_parquet(tmp_path / "ref.parquet", df),
        )

    def test_different_values_raise(self, tmp_path: Path) -> None:
        """Frame inequality (AssertionError from pandas) is wrapped as ValidationFailed."""
        with pytest.raises(ValidationFailed):
            validate_parquet(
                _write_parquet(tmp_path / "test.parquet", pd.DataFrame({"a": [1, 2, 3]})),
                _write_parquet(tmp_path / "ref.parquet", pd.DataFrame({"a": [1, 2, 999]})),
            )

    def test_multichar_coercion_failure_raises_validation_failed(self, tmp_path: Path) -> None:
        """ValueError from _convert_string_to_int is wrapped as ValidationFailed."""
        with pytest.raises(ValidationFailed):
            validate_parquet(
                _write_parquet(
                    tmp_path / "test.parquet",
                    pd.DataFrame({"col": pd.array([65, 66], dtype=np.int8)}),
                ),
                _write_parquet(tmp_path / "ref.parquet", pd.DataFrame({"col": ["AB", "CD"]})),
            )

    def test_missing_reference_file_propagates(self, tmp_path: Path) -> None:
        """FileNotFoundError (not a data-mismatch) propagates unchanged — bugs stay loud."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        test_path = _write_parquet(tmp_path / "test.parquet", df)
        with pytest.raises(FileNotFoundError):
            validate_parquet(test_path, tmp_path / "does_not_exist.parquet")

    def test_int8_vs_string_coercion(self, tmp_path: Path) -> None:
        validate_parquet(
            _write_parquet(
                tmp_path / "test.parquet", pd.DataFrame({"col": pd.array([65, 66], dtype=np.int8)})
            ),
            _write_parquet(tmp_path / "ref.parquet", pd.DataFrame({"col": ["A", "B"]})),
        )

    def test_wider_int_promotes_narrower(self, tmp_path: Path) -> None:
        validate_parquet(
            _write_parquet(
                tmp_path / "test.parquet", pd.DataFrame({"col": pd.array([1, 2], dtype=np.int64)})
            ),
            _write_parquet(
                tmp_path / "ref.parquet", pd.DataFrame({"col": pd.array([1, 2], dtype=np.int32)})
            ),
        )

    def test_float_coercion(self, tmp_path: Path) -> None:
        validate_parquet(
            _write_parquet(
                tmp_path / "test.parquet",
                pd.DataFrame({"col": pd.array([1.0, 2.0], dtype=np.float64)}),
            ),
            _write_parquet(
                tmp_path / "ref.parquet",
                pd.DataFrame({"col": pd.array([1.0, 2.0], dtype=np.float32)}),
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
