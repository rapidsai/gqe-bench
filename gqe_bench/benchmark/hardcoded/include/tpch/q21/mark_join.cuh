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

#include <tpch/q21/task.hpp>
#include <utility/mark_join.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>

namespace gqe_python {
namespace benchmark {
namespace q21 {

template <typename T>
using mark_join_map_type = gqe_python::utility::multimap_type<T, cudf::size_type>;
template <typename T>
using mark_join_map_ref_type = typename mark_join_map_type<
  T>::template ref_type<cuco::op::find_tag, cuco::op::for_each_tag, cuco::op::insert_tag>;

// Assume the join keys are never max identifier.
template <typename T>
constexpr cuco::empty_key<T> empty_key_sentinel(
  gqe_python::utility::unset_mark(std::numeric_limits<T>::max()));
// Row indices are non-negative in nature.
constexpr cuco::empty_value<cudf::size_type> empty_value_sentinel(-1);

/**
 * @brief Left anti join parameters.
 * @note The parameters are wrapped inside a struct because the
 * parameters are long in symbols, which causes compile error.
 *
 * @tparam Identifier The type of the identifier (e.g., int32_t, int64_t).
 */
template <typename Identifier>
struct left_anti_join_params {
  mark_join_map_ref_type<Identifier> map;  ///< Hash map used for mark join
  gqe_python::utility::bloom_filter_ref_type<Identifier>
    bloom_filter;                                     ///< Bloom filter for early rejection
  cudf::column_device_view build_suppkey_column;      ///< Build side suppkey column
  cudf::column_device_view probe_receiptdate_column;  ///< Probe side receiptdate column
  cudf::column_device_view probe_commitdate_column;   ///< Probe side commitdate column
  cudf::column_device_view probe_suppkey_column;      ///< Probe side suppkey column
  cudf::column_device_view probe_orderkey_column;     ///< Probe side orderkey column
  cudf::size_type* d_counter;                         ///< Device counter for output rows
};

/**
 * @brief Left anti join parameters.
 * @tparam Identifier The type of the identifier (e.g., int32_t, int64_t).
 */
template <typename Identifier>
struct left_semi_join_params {
  mark_join_map_ref_type<Identifier> map;  ///< Hash map used for mark join
  gqe_python::utility::bloom_filter_ref_type<Identifier>
    bloom_filter;                                  ///< Bloom filter for early rejection
  cudf::column_device_view build_suppkey_column;   ///< Build side suppkey column
  cudf::column_device_view probe_suppkey_column;   ///< Probe side suppkey column
  cudf::column_device_view probe_orderkey_column;  ///< Probe side orderkey column
  cudf::size_type* d_counter;                      ///< Device counter for output rows
};

/**
 * @brief Left anti join kernel for TPC-H Q21.
 *
 * Performs left anti join operation with the following logic:
 * - Build hash map on left table using orderkey.
 * - Probe with right table rows that satisfy: receiptdate > commitdate.
 * - Mark entries where: orderkey matches AND suppkey differs.
 *
 * @param[in, out] params The parameters for left_anti_join_params struct containing:
 *   - map[in]: Hash map built from left table orderkey
 *   - bloom_filter[in]: Bloom filter for early rejection
 *   - build_suppkey_column[in]: Left table suppkey column
 *   - probe_receiptdate_column[in]: Right table receiptdate column
 *   - probe_commitdate_column[in]: Right table commitdate column
 *   - probe_suppkey_column[in]: Right table suppkey column
 *   - probe_orderkey_column[in]: Right table orderkey column
 *   - d_counter[out]: Device counter for marked entries
 */
template <typename Identifier>
__global__ void left_anti_join_kernel(left_anti_join_params<Identifier> params);

/**
 * @brief Left semi join kernel for TPC-H Q21.
 *
 * @param[in, out] params The parameters for the left semi join kernel containing:
 *   - map[in]: Hash map built from left table orderkey
 *   - bloom_filter[in]: Bloom filter for early rejection
 *   - build_suppkey_column[in]: Left table suppkey column
 *   - probe_suppkey_column[in]: Right table suppkey column
 *   - probe_orderkey_column[in]: Right table orderkey column
 *   - d_counter[out]: Device counter for marked entries
 */
template <typename Identifier>
__global__ void left_semi_join_kernel(left_semi_join_params<Identifier> params);

/**
 * @brief Iterate over the join map and output the customer indices (uunmarked entries).
 *
 * @param[in] map_device_view The device view of the join map.
 * @param[out] out_indices The output indices for the customers.
 * @param[out] d_counter The global counter for the number of output rows.
 */
template <bool IsAntiJoin, typename Identifier>
__global__ void iterate_join_map(mark_join_map_ref_type<Identifier> map_device_view,
                                 cudf::size_type* out_indices,
                                 cudf::size_type* d_counter);

}  // namespace q21
}  // namespace benchmark
}  // namespace gqe_python
