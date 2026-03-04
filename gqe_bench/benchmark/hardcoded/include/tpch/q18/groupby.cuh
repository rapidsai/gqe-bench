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
