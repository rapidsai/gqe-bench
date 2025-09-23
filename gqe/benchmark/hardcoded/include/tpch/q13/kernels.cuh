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

#include <utility/config.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>

namespace gqe_python {
namespace benchmark {
namespace q13 {

template <typename identifier_type>

/**
 * @brief Filter kernel for the orders relation.
 *
 * @pre d_global_offset must be initialized with 0.
 *
 * @param[in] o_custkey The respective input column.
 * @param[in] o_comment The respective input column.
 * @param[out] d_global_offset A device-wide offset counter for the result rows.
 * @param[in] out_o_custkey The respective output column.
 */
__global__ void filter_orders_kernel(cudf::column_device_view o_custkey,
                                     cudf::column_device_view o_comment,
                                     cudf::size_type* d_global_offset,
                                     cudf::mutable_column_device_view out_o_custkey);

/**
 * @brief Groupjoin probe kernel.
 *
 * Performs a left outer join on customer and orders, and group by on c_custkey.
 *
 * @pre The counters in customer_map must be initialized to 0.
 * @pre d_global_offset must be initialized to 0.
 *
 * @param[in,out] customer_map Hash map containing `c_custkey` and a counter.
 * @param[in] o_custkey Orders table customer foreign key.
 */
template <typename identifier_type>
__global__ void groupjoin_probe_kernel(
  utility::map_ref_type<identifier_type, cudf::size_type> customer_map,
  cudf::column_device_view o_custkey);

/**
 * @brief Fused orders filter and groupjoin probe kernel.
 *
 * @pre The counters in customer_map must be initialized to 0.
 * @pre d_global_offset must be initialized to 0.
 *
 * @param[in,out] customer_map Hash map containing `c_custkey` and a counter.
 * @param[in] o_custkey Orders table customer foreign key.
 * @param[in] o_comment Orders table comment column.
 */
template <typename identifier_type>
__global__ void fused_filter_probe_kernel(
  utility::map_ref_type<identifier_type, cudf::size_type> customer_map,
  cudf::column_device_view o_custkey,
  cudf::column_device_view o_comment);

/**
 * @brief Groupjoin retrieve kernel.
 *
 * @pre d_global_offset must be initialized to 0.
 *
 * @param[in,out] customer_map Hash map containing `c_custkey` and a counter.
 * @param[out] d_global_offset A device-wide offset counter for the result rows.
 * @param[out] out_c_custkey Output column with the group key.
 * @param[out] out_c_count Output column with the count of non-NULL orders.
 */
template <typename identifier_type>
__global__ void groupjoin_retrieve_kernel(
  utility::map_ref_type<identifier_type, cudf::size_type> customer_map,
  cudf::size_type* d_global_offset,
  cudf::mutable_column_device_view out_c_custkey,
  cudf::mutable_column_device_view out_c_count);

}  // namespace q13
}  // namespace benchmark
}  // namespace gqe_python
