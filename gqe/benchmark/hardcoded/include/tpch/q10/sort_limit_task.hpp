/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
