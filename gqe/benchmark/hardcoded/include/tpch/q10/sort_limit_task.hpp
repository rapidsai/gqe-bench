/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <gqe/physical/relation.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe_python {
namespace benchmark {
namespace q10 {

/**
 * @brief Sort input table and output only the first `limit` rows.
 *
 * @param[in] input_table The input table.
 * @param[in] key_col_idx The column index of the sort key.
 * @param[in] limit Number of output rows.
 * @param[in] projection_indices Indices of columns to be included in output.
 */
std::shared_ptr<gqe::physical::relation> sort_limit(
  std::shared_ptr<gqe::physical::relation> input_table,
  const cudf::size_type key_col_idx,
  const cudf::size_type limit,
  std::vector<cudf::size_type> projection_indices);

}  // namespace q10
}  // namespace benchmark
}  // namespace gqe_python
