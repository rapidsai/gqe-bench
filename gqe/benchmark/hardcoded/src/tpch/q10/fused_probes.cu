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

#include <tpch/q10/fused_probes.cuh>
#include <utility/config.hpp>
#include <utility/write_buffer.cuh>

#include <gqe/utility/helpers.hpp>

#include <cuda/atomic>
#include <cuda/std/utility>

#include <cstdint>

namespace gqe_python {
namespace benchmark {
namespace q10 {

template <typename Identifier, typename MultimapRef, typename BloomFilterRef, typename MapRef>
__global__ void fused_probes_kernel(MultimapRef o_custkey_multimap,
                                    BloomFilterRef bloom_filter,
                                    MapRef n_nationkey_map,
                                    cudf::column_device_view c_custkey_column,
                                    cudf::column_device_view c_nationkey_column,
                                    cudf::size_type* d_global_offset,
                                    fused_probes_output out_indices)
{
  auto const row_count   = c_custkey_column.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const loop_end =
    gqe::utility::divide_round_up(row_count, utility::warp_size) * utility::warp_size;

  fused_probes_op fpo(o_custkey_multimap, n_nationkey_map);

  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_offset_ref(*d_global_offset);

  __shared__
    typename utility::write_buffer_op<cudf::size_type, cudf::size_type, cudf::size_type>::storage_t
      wbs;

  utility::write_buffer_op<cudf::size_type, cudf::size_type, cudf::size_type> wb(
    &wbs,
    global_offset_ref,
    out_indices.join_orders_lineitem_table_indices.data<cudf::size_type>(),
    out_indices.nation_table_indices.data<cudf::size_type>(),
    out_indices.customer_table_indices.data<cudf::size_type>());

  for (cudf::size_type idx = loop_begin; idx < loop_end; idx += loop_stride) {
    __syncwarp();
    bool is_active = idx < row_count;

    auto c_custkey = c_custkey_column.element<Identifier>(idx);
    is_active &= bloom_filter.contains(c_custkey);
    // guard against possible divergence from Bloom filter
    __syncwarp();

    fpo.fused_probes(
      c_custkey, idx, c_nationkey_column, is_active, [&wb](auto const& slot) { wb.write(slot); });
  }
  wb.flush();
}

template __global__ void fused_probes_kernel<int32_t>(
  utility::multimap_ref_type<int32_t, cudf::size_type> o_custkey_multimap,
  utility::bloom_filter_ref_type<int32_t, cuco::identity_hash<int32_t>> bloom_filter,
  utility::map_ref_type<int32_t, cudf::size_type, cuco::identity_hash<int32_t>> n_nationkey_map,
  cudf::column_device_view c_custkey_column,
  cudf::column_device_view c_nationkey_column,
  cudf::size_type* d_global_offset,
  fused_probes_output out_indices);

template __global__ void fused_probes_kernel<int64_t>(
  utility::multimap_ref_type<int64_t, cudf::size_type> o_custkey_multimap,
  utility::bloom_filter_ref_type<int64_t, cuco::identity_hash<int64_t>> bloom_filter,
  utility::map_ref_type<int64_t, cudf::size_type, cuco::identity_hash<int64_t>> n_nationkey_map,
  cudf::column_device_view c_custkey_column,
  cudf::column_device_view c_nationkey_column,
  cudf::size_type* d_global_offset,
  fused_probes_output out_indices);

}  // namespace q10
}  // namespace benchmark
}  // namespace gqe_python
