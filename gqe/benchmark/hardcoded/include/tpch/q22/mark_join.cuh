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
