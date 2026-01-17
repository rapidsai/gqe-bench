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

#include <rmm/cuda_stream_view.hpp>

namespace gqe_python {
namespace benchmark {
namespace q18 {

/**
 * @brief CUDA kernel for retrieving group by results with filtering.
 *
 * This kernel iterates through the hash map and retrieves entries where
 * the aggregated quantity is greater than 300.0.
 *
 * @param[in] hash_map_ref The hash map reference for iteration (uses Q18-specific bucket_size=1)
 * @param[in,out] d_global_offset Global offset for output indexing
 * @param[out] out_l_orderkey Output column for order keys
 * @param[out] out_sum_quantity Output column for aggregated quantities
 */
template <typename Identifier>
__global__ void groupby_retrieve_kernel(utility::q18::map_ref_type<Identifier, double> hash_map_ref,
                                        cudf::size_type* d_global_offset,
                                        cudf::mutable_column_device_view out_l_orderkey,
                                        cudf::mutable_column_device_view out_sum_quantity);

}  // namespace q18
}  // namespace benchmark
}  // namespace gqe_python
