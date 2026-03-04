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

namespace gqe_python {
namespace benchmark {
namespace q10 {

/**
 * @brief Build hash map for unique-key inner join.
 *
 * @param[in] build_side_table The build-side table.
 * @param[in] key_column_idx Index of key column in build-side table.
 * @param[in] enable_bloom_filter Flag for building a Bloom filter to reduce hash map lookup during
 * probing.
 */
std::shared_ptr<gqe::physical::relation> unique_key_inner_join_build(
  std::shared_ptr<gqe::physical::relation> build_side_table,
  const cudf::size_type key_column_idx,
  bool enable_bloom_filter);

/**
 * @brief Probe hash map for unique-key inner join.
 *
 * @param[in] build_side_map The build-side hash map.
 * @param[in] build_side_table The build-side table.
 * @param[in] probe_side_table The probe-side table.
 * @param[in] probe_side_key_column_idx Index of key column in probe-side table.
 * @param[in] projection_indices Indices of projected columns.
 */
std::shared_ptr<gqe::physical::relation> unique_key_inner_join_probe(
  std::shared_ptr<gqe::physical::relation> build_side_map,
  std::shared_ptr<gqe::physical::relation> build_side_table,
  std::shared_ptr<gqe::physical::relation> probe_side_table,
  cudf::size_type build_side_key_column_idx,
  cudf::size_type probe_side_key_column_idx,
  std::vector<cudf::size_type>& projection_indices);

}  // namespace q10
}  // namespace benchmark
}  // namespace gqe_python
