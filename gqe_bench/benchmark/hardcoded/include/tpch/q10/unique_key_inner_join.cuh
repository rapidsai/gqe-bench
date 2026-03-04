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

namespace gqe_python {
namespace benchmark {
namespace q10 {

// struct for passing output columns to kernel
// This is a workaround to a compiler issue where heavily templated data structures result in long
// symbol names that lead to mismatched function declaration/definition in cudafe-generated code. In
// `unique_key_inner_join_probe_kernel`, the parameters with long string representations are
// `build_side_map` and `bloom_filter`, but for some reason this particular issue only manifests
// when two (vs. just one) parameters of the same type are also part of the kernel function
// signature. Packing the two `mutable_column_device_view` parameters into a single struct resolves
// the issue.
struct unique_key_inner_join_probe_output {
  cudf::mutable_column_device_view build_side_indices;
  cudf::mutable_column_device_view probe_side_indices;
};

/**
 * @brief Unique-key inner join probe kernel
 *
 * Probe the build-side hash map with the probe keys and output the row indices
 * of the matches.
 *
 * @param[in] build_side_map The build-side hash map.
 * @param[in] bloom_filter The Bloom filter built with build-side keys.
 * @param[in] use_bloom_filter Flag indicating whether the Bloom filter should be used.
 * @param[in] probe_side_key_column The probe-side key column.
 * @param[in] l_returnflag Hardcoded column for Q10 filter on lineitem.
 * @param[in, out] d_global_offset Global counter for number of output rows.
 * @param[out] out_indices Struct with build-side and probe-side row indices of matches.
 */
template <typename Identifier, typename MapRef, typename BloomFilterRef>
__global__ void unique_key_inner_join_probe_kernel(MapRef build_side_map,
                                                   BloomFilterRef bloom_filter,
                                                   cudf::column_device_view probe_side_key_column,
                                                   cudf::column_device_view l_returnflag,
                                                   cudf::size_type* d_global_offset,
                                                   unique_key_inner_join_probe_output out_indices);

}  // namespace q10
}  // namespace benchmark
}  // namespace gqe_python
