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
