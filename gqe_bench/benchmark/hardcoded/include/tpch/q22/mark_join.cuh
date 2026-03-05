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

#include <tpch/q22/task.hpp>
#include <utility/mark_join.cuh>
#include <utility/write_buffer.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>

namespace gqe_python {
namespace benchmark {
namespace q22 {

template <typename T>
using mark_join_map_type = gqe_python::utility::multimap_type<T, cudf::size_type>;
template <typename T>
using mark_join_map_ref_type =
  typename mark_join_map_type<T>::template ref_type<cuco::op::find_tag, cuco::op::for_each_tag>;

// Assume the join keys are never max identifier.
template <typename T>
constexpr cuco::empty_key<T> empty_key_sentinel(
  gqe_python::utility::unset_mark(std::numeric_limits<T>::max()));
// Row indices are non-negative in nature.
constexpr cuco::empty_value<cudf::size_type> empty_value_sentinel(-1);

/**
 * @brief Mark join probing kernel.
 *
 * @param[in] customer_map hash map used for mark join.
 * @param[in] bloom_filter bloom filter used for early rejection.
 * @param[in] o_custkey_column output column for customer table c_custkey column.
 * @param[out] d_counter output global counter for number of output rows.
 */
template <typename identifier_type>
__global__ void mark_join_kernel(
  mark_join_map_ref_type<identifier_type> customer_map,
  gqe_python::utility::bloom_filter_ref_type<identifier_type> bloom_filter,
  cudf::column_device_view o_custkey_column,
  cudf::size_type* d_counter);

/**
 * @brief Iterate over the join map and output the customer indices (uunmarked entries).
 *
 * @param[in] map_device_view The device view of the join map.
 * @param[out] customer_out_indices The output indices for the customers.
 * @param[out] d_counter_ref The global counter for the number of output rows.
 */
template <typename identifier_type>
__global__ void iterate_join_map(mark_join_map_ref_type<identifier_type> map_device_view,
                                 cudf::size_type* customer_out_indices,
                                 cudf::size_type* d_counter_ref);

}  // namespace q22
}  // namespace benchmark
}  // namespace gqe_python
