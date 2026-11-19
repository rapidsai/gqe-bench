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

"""Query result validation via parquet comparison."""

import decimal
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from pandas.testing import assert_series_equal

# Absolute tolerance for FLOAT64 columns.
_FLOAT_ATOL = 1e-6

# Absolute tolerance once any column is DECIMAL. Mirrors ABS_DECIMAL_TOLERANCE in
# gqe/test/end_to_end/verify_parquet.py.
_DECIMAL_ATOL = 1e-2


class ValidationFailed(RuntimeError):
    """Output parquet does not match the reference.

    Raised specifically for data-mismatch conditions: comparison inequality
    (AssertionError from pandas) and expected-type-coercion failure
    (ValueError from _convert_string_to_int). Other exceptions from
    validate_parquet — missing files, I/O errors, pandas/pyarrow internal
    bugs — propagate unchanged so they surface as real issues rather than
    being laundered into a recoverable "validation failed" signal.
    """


def _convert_string_to_int(df: pd.DataFrame, col: str) -> None:
    """Convert single-char ASCII string column to int8."""
    if not df[col].apply(lambda x: len(x) == 1).all():
        raise ValueError("Can only convert single-char (ASCII) strings to INT8 type")
    df[col] = df[col].apply(lambda x: ord(x)).astype(np.int8)


def _normalize_type(source: pd.DataFrame, target: pd.DataFrame, col: str) -> None:
    """Cast target column to match source column's dtype."""
    target[col] = target[col].astype(source[col].dtype)


def _is_decimal_series(s: pd.Series) -> bool:
    """True if ``s`` is an object-dtype Series holding ``decimal.Decimal`` values.

    pandas/pyarrow surface a parquet DECIMAL column as object dtype containing
    ``decimal.Decimal`` instances; there is no numeric dtype for them, so dtype
    alone cannot answer this.
    """
    if s.dtype != object:
        return False
    try:
        first_valid_value = s.loc[s.first_valid_index()]
    except (KeyError, TypeError):
        return False
    return isinstance(first_valid_value, decimal.Decimal)


def _to_decimal(value: object) -> decimal.Decimal:
    """Coerce a scalar to Decimal, routing floats through str to avoid binary artifacts."""
    if isinstance(value, decimal.Decimal):
        return value
    if isinstance(value, (np.integer, int)):
        return decimal.Decimal(int(value))
    if isinstance(value, (np.floating, float)):
        return decimal.Decimal(str(float(value)))
    return decimal.Decimal(str(value))


def _assert_decimal_series_equal(lhs: pd.Series, rhs: pd.Series, atol: float) -> None:
    """Compare two decimal Series elementwise within ``atol``.

    ``assert_series_equal`` cannot apply a tolerance to object-dtype decimals, so
    the comparison is done in ``Decimal`` arithmetic here.
    """
    if len(lhs) != len(rhs):
        raise AssertionError(f"Decimal series length mismatch: left={len(lhs)}, right={len(rhs)}")
    threshold = decimal.Decimal(str(atol))
    for i, (lv, rv) in enumerate(zip(lhs.values, rhs.values)):
        if pd.isna(lv) and pd.isna(rv):
            continue
        if pd.isna(lv) != pd.isna(rv):
            raise AssertionError(f"Decimal null mismatch at index {i}: left={lv}, right={rv}")
        if abs(_to_decimal(lv) - _to_decimal(rv)) > threshold:
            raise AssertionError(
                f"Decimal mismatch at index {i}: left={lv}, right={rv}, atol={atol}"
            )


def _assert_frame_equal_decimal_aware(lhs: pd.DataFrame, rhs: pd.DataFrame, atol: float) -> None:
    """Assert frame equality, comparing decimal columns elementwise."""
    if not lhs.columns.equals(rhs.columns):
        raise AssertionError(f"Column mismatch: left={list(lhs.columns)} right={list(rhs.columns)}")
    if not lhs.index.equals(rhs.index):
        raise AssertionError("Index mismatch between compared DataFrames.")
    for col in lhs.columns:
        if _is_decimal_series(lhs[col]) or _is_decimal_series(rhs[col]):
            _assert_decimal_series_equal(lhs[col], rhs[col], atol=atol)
        else:
            assert_series_equal(lhs[col], rhs[col], atol=atol)


def _restore_q11_total_order(
    query_name: str, df_test: pd.DataFrame, df_ref: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort Q11's frames by ps_partkey so they can be compared positionally.

    DELETE THIS, its call in _compare, and the query_name argument threaded in
    from runner.py, once GQE supports DECIMAL natively.

    Q11 is ``ORDER BY value DESC`` with no tiebreaker, so rows sharing a
    ``value`` may appear in any order. ``value`` is
    ``sum(ps_supplycost * ps_availqty)``; with ps_supplycost carried as float64
    instead of DECIMAL(15,2), rows equal at DECIMAL precision differ in the low
    digits and sort differently from the decimal-exact reference. Comparing
    positionally then reports a mismatch on ps_partkey for a correct result.

    ps_partkey is Q11's GROUP BY key and therefore unique per row, so sorting
    both frames by it gives the same total order on each side. ``value`` is
    rounded to the two fractional digits TPC-H specifies for DECIMAL.

    Returns the frames unchanged for every other query.
    """
    if query_name.lstrip("qQ").split("_")[0] != "11":
        return df_test, df_ref
    partkey, value = df_ref.columns[0], df_ref.columns[1]
    df_test = df_test.sort_values(by=partkey).reset_index(drop=True)
    df_test[value] = df_test[value].round(2)
    return df_test, df_ref.sort_values(by=partkey).reset_index(drop=True)


def _compare(test_file: Path, ref_file: Path, query_name: str) -> None:
    """Read both parquet files, apply type coercion, assert equality."""
    df_test = pd.read_parquet(test_file)
    df_ref = pd.read_parquet(ref_file)

    df_test.columns = df_ref.columns

    for col in df_test.columns:
        if df_test[col].dtype == np.int8 and is_string_dtype(df_ref[col]):
            _convert_string_to_int(df_ref, col)
        elif not is_numeric_dtype(df_test[col]) and is_numeric_dtype(df_ref[col]):
            _normalize_type(df_ref, df_test, col)
        elif is_float_dtype(df_test[col]) and not is_integer_dtype(df_ref[col]):
            _normalize_type(df_test, df_ref, col)
        elif is_numeric_dtype(df_test[col]) and is_numeric_dtype(df_ref[col]):
            test_size = df_test[col].dtype.itemsize
            ref_size = df_ref[col].dtype.itemsize
            if test_size > ref_size:
                _normalize_type(df_test, df_ref, col)
            elif test_size < ref_size:
                _normalize_type(df_ref, df_test, col)

    df_test, df_ref = _restore_q11_total_order(query_name, df_test, df_ref)

    # One tolerance for the whole frame: a decimal column anywhere means the run
    # carries DECIMAL, and the looser bound applies to every column of it.
    has_decimal = any(
        _is_decimal_series(df_test[col]) or _is_decimal_series(df_ref[col])
        for col in df_test.columns
    )
    if has_decimal:
        _assert_frame_equal_decimal_aware(df_test, df_ref, atol=_DECIMAL_ATOL)
    else:
        pd.testing.assert_frame_equal(df_test, df_ref, atol=_FLOAT_ATOL)


def validate_parquet(test_file: Path, ref_file: Path, query_name: str = "") -> None:
    """Compare a query result parquet file against a reference file.

    Performs type coercion to handle expected type mismatches between
    GQE output and reference data, then asserts frame equality.

    ``query_name`` selects the Q11 workaround in _restore_q11_total_order and
    is otherwise unused.

    Raises ValidationFailed specifically on:
      * AssertionError — frames differ.
      * ValueError — type coercion impossible (_convert_string_to_int).
    Other exceptions (FileNotFoundError, OSError, pandas/pyarrow internals)
    propagate as-is.
    """
    try:
        _compare(test_file, ref_file, query_name)
    except (AssertionError, ValueError) as e:
        raise ValidationFailed(str(e)) from e
