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

#include <tpch/q10/unique_key_inner_join.cuh>
#include <utility/config.hpp>
#include <utility/unique_key_inner_join.cuh>
#include <utility/write_buffer.cuh>

#include <gqe/utility/helpers.hpp>

#include <cuda/atomic>
#include <cuda/std/utility>

#include <cstdint>

namespace gqe_python {
namespace benchmark {
namespace q10 {

template <typename Identifier, typename MapRef, typename BloomFilterRef>
__global__ void unique_key_inner_join_probe_kernel(MapRef build_side_map,
                                                   BloomFilterRef bloom_filter,
                                                   cudf::column_device_view probe_side_key_column,
                                                   cudf::column_device_view l_returnflag,
                                                   cudf::size_type* d_global_offset,
                                                   unique_key_inner_join_probe_output out_indices)
{
  auto const row_count   = probe_side_key_column.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const loop_end =
    gqe::utility::divide_round_up(row_count, utility::warp_size) * utility::warp_size;

  utility::unique_key_inner_join_op ukijo(build_side_map);

  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_offset_ref(*d_global_offset);

  __shared__ typename utility::write_buffer_op<cudf::size_type, cudf::size_type>::storage_t wbs;

  utility::write_buffer_op<cudf::size_type, cudf::size_type> wb(
    &wbs,
    global_offset_ref,
    out_indices.build_side_indices.data<cudf::size_type>(),
    out_indices.probe_side_indices.data<cudf::size_type>());

  for (cudf::size_type idx = loop_begin; idx < loop_end; idx += loop_stride) {
    __syncwarp();
    bool is_active = idx < row_count;
    // hardcoded Q10 filter on lineitem: l_returnflag = 'R'
    if (is_active) is_active = (l_returnflag.element<char>(idx) == 82 /* ASCII code for 'R' */);

    auto key = is_active ? probe_side_key_column.element<Identifier>(idx) : Identifier{};
    if (bloom_filter.block_extent()) is_active &= bloom_filter.contains(key);
    // guard against possible divergence from Bloom filter
    __syncwarp();
    auto join_predicate = [&](auto const& entry) -> bool { return key == entry.first; };

    ukijo.probe(&key, idx, join_predicate, is_active, [&wb](auto const& slot) { wb.write(slot); });
  }
  wb.flush();
}

template __global__ void unique_key_inner_join_probe_kernel<int32_t>(
  utility::map_ref_type<int32_t, cudf::size_type, cuco::identity_hash<int32_t>> build_side_map,
  utility::bloom_filter_ref_type<int32_t, cuco::identity_hash<int32_t>> bloom_filter,
  cudf::column_device_view probe_side_key_column,
  cudf::column_device_view l_returnflag,
  cudf::size_type* d_global_offset,
  unique_key_inner_join_probe_output out_indices);

template __global__ void unique_key_inner_join_probe_kernel<int64_t>(
  utility::map_ref_type<int64_t, cudf::size_type, cuco::identity_hash<int64_t>> build_side_map,
  utility::bloom_filter_ref_type<int64_t, cuco::identity_hash<int64_t>> bloom_filter,
  cudf::column_device_view probe_side_key_column,
  cudf::column_device_view l_returnflag,
  cudf::size_type* d_global_offset,
  unique_key_inner_join_probe_output out_indices);

}  // namespace q10
}  // namespace benchmark
}  // namespace gqe_python
