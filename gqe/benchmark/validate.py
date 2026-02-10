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

from collections.abc import Callable
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)


def convert_string_to_int(df1: pd.DataFrame, col: str):
    all_single_char = df1[col].apply(lambda x: len(x) == 1).all()
    if not all_single_char:
        raise Exception("Can only convert single-char (ASCII) strings to INT8 type")
    df1[col] = df1[col].apply(lambda x: ord(x) if isinstance(x, str) and len(x) == 1 else None)
    df1[col] = df1[col].astype(np.int8)


def normalize_type(df1: pd.DataFrame, df2: pd.DataFrame, col: str):
    new_type = df1[col].dtypes.name
    df2[col] = df2[col].astype(new_type)


def validate_parquet(
    test_file: str,
    ref_file: str,
    validator: Optional[Callable[[pd.DataFrame, pd.DataFrame, float], None]],
):
    df_gqe = pd.read_parquet(test_file)
    df_ref = pd.read_parquet(ref_file)

    # normalize column names
    df_gqe.columns = df_ref.columns
    # normalize column types
    for col in df_gqe.columns:
        if df_gqe[col].dtype == np.int8 and is_string_dtype(df_ref[col]):
            convert_string_to_int(df_ref, col)
        elif not is_numeric_dtype(df_gqe[col]) and is_numeric_dtype(df_ref[col]):
            # if only one column is numeric, covert to numeric type
            normalize_type(df_ref, df_gqe, col)
        elif is_float_dtype(df_gqe[col]) and not is_integer_dtype(df_ref[col]):
            # if only one floating point
            normalize_type(df_gqe, df_ref, col)
        elif is_numeric_dtype(df_gqe[col]) and is_numeric_dtype(df_ref[col]):
            # if both are numeric convert to larger type
            gqe_col_item_size = df_gqe[col].dtype.itemsize
            ref_col_item_size = df_ref[col].dtype.itemsize
            if gqe_col_item_size > ref_col_item_size:
                normalize_type(df_gqe, df_ref, col)
            elif gqe_col_item_size < ref_col_item_size:
                normalize_type(df_ref, df_gqe, col)

    # Validate that GQE result is the same as the reference result
    validator(df_gqe, df_ref, 1e-06)
