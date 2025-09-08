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
#include <cudf/copying.hpp>

namespace gqe_python {
namespace benchmark {
namespace q22 {

/**
 * @brief Fused kernel for projection and filter operations.
 *
 * @param[in] c_phone_column customer table c_phone column.
 * @param[in] c_acctbal_column customer table c_acctbal column.
 * @param[in] c_custkey_column customer table c_custkey column.
 * @param[in, out] d_counter global counter for number of output rows.
 * @param[out] out_c_phone_column output column for customer table c_phone column.
 * @param[out] out_c_acctbal_column output column for customer table c_acctbal column.
 * @param[out] out_c_custkey_column output column for customer table c_custkey column.
 */
template <typename identifier_type, typename decimal_type>
__global__ void fused_project_filter_kernel(cudf::column_device_view c_phone_column,
                                            cudf::column_device_view c_acctbal_column,
                                            cudf::column_device_view c_custkey_column,
                                            cudf::size_type* d_counter,
                                            char* out_c_phone_column,
                                            cudf::mutable_column_device_view out_c_acctbal_column,
                                            cudf::mutable_column_device_view out_c_custkey_column);
/**
 * @brief Create a customized cudf string column from a device vector of characters, each string
 * contains two chars. Construct the offset vector inside this function.
 *
 * @param[in] d_chars Device vector of characters.
 * @param[in] num Number of characters.
 */
std::unique_ptr<cudf::column> make_customized_string_column(rmm::device_uvector<char> d_chars,
                                                            cudf::size_type num);

}  // namespace q22
}  // namespace benchmark
}  // namespace gqe_python