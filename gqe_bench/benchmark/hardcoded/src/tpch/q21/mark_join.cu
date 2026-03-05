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

#include <tpch/q21/mark_join.cuh>
#include <utility/write_buffer.cuh>

namespace gqe_python {
namespace benchmark {
namespace q21 {

template <typename Identifier>
__global__ void left_anti_join_kernel(left_anti_join_params<Identifier> params)
{
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> d_global_offset(*(params.d_counter));
  gqe_python::utility::mark_join_op<mark_join_map_ref_type<Identifier>> mjo(params.map,
                                                                            d_global_offset);
  mjo.pre_set_mark_loop();
  auto const row_count   = params.probe_orderkey_column.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const loop_end = gqe::utility::divide_round_up(row_count, gqe_python::utility::warp_size) *
                        gqe_python::utility::warp_size;

  for (cudf::size_type idx = loop_begin; idx < loop_end; idx += loop_stride) {
    __syncwarp();
    bool is_active = idx < row_count;
    // Fuse the filter here.
    if (is_active) {
      is_active = params.probe_receiptdate_column.template element<cudf::timestamp_D>(idx) >
                  params.probe_commitdate_column.template element<cudf::timestamp_D>(idx);
    }
    Identifier orderkey =
      is_active ? params.probe_orderkey_column.template element<Identifier>(idx) : -1;
    if (is_active) { is_active = params.bloom_filter.contains(orderkey); }
    Identifier suppkey =
      is_active ? params.probe_suppkey_column.template element<Identifier>(idx) : -1;
    auto join_predicate = [&](auto const& entry) -> bool {
      return entry.first == orderkey &&
             params.build_suppkey_column.template element<Identifier>(entry.second) != suppkey;
    };
    mjo.template set_mark<false>(&orderkey, join_predicate, is_active);
  }

  mjo.template post_set_mark_loop<false>();
}

template <typename Identifier>
__global__ void left_semi_join_kernel(left_semi_join_params<Identifier> params)
{
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> d_global_offset(*(params.d_counter));
  gqe_python::utility::mark_join_op<mark_join_map_ref_type<Identifier>> mjo(params.map,
                                                                            d_global_offset);
  mjo.pre_set_mark_loop();
  auto const row_count   = params.probe_orderkey_column.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const loop_end = gqe::utility::divide_round_up(row_count, gqe_python::utility::warp_size) *
                        gqe_python::utility::warp_size;

  for (cudf::size_type idx = loop_begin; idx < loop_end; idx += loop_stride) {
    __syncwarp();
    bool is_active = idx < row_count;
    Identifier orderkey =
      is_active ? params.probe_orderkey_column.template element<Identifier>(idx) : -1;
    if (is_active) { is_active = params.bloom_filter.contains(orderkey); }
    Identifier suppkey =
      is_active ? params.probe_suppkey_column.template element<Identifier>(idx) : -1;
    auto join_predicate = [&](auto const& entry) -> bool {
      return entry.first == orderkey &&
             params.build_suppkey_column.template element<Identifier>(entry.second) != suppkey;
    };
    mjo.template set_mark<false>(&orderkey, join_predicate, is_active);
  }

  mjo.template post_set_mark_loop<false>();
}

template <bool IsAntiJoin, typename Identifier>
__global__ void iterate_join_map(mark_join_map_ref_type<Identifier> map_device_view,
                                 cudf::size_type* out_indices,
                                 cudf::size_type* d_counter)
{
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> d_global_offset{*d_counter};

  __shared__ typename utility::write_buffer_op<cudf::size_type>::storage_t wbs;
  utility::write_buffer_op<cudf::size_type> wb(&wbs, d_global_offset, out_indices);

  gqe_python::utility::mark_join_op<mark_join_map_ref_type<Identifier>> mjo(map_device_view,
                                                                            d_global_offset);
  mjo.template retrieve<IsAntiJoin>([&wb](auto const& slot) { wb.write(slot); });

  wb.flush();
}

template __global__ void left_anti_join_kernel<int32_t>(left_anti_join_params<int32_t> params);

template __global__ void left_anti_join_kernel<int64_t>(left_anti_join_params<int64_t> params);

template __global__ void left_semi_join_kernel<int32_t>(left_semi_join_params<int32_t> params);

template __global__ void left_semi_join_kernel<int64_t>(left_semi_join_params<int64_t> params);

template __global__ void iterate_join_map<true, int32_t>(
  mark_join_map_ref_type<int32_t> map_device_view,
  cudf::size_type* out_indices,
  cudf::size_type* d_counter);

template __global__ void iterate_join_map<false, int32_t>(
  mark_join_map_ref_type<int32_t> map_device_view,
  cudf::size_type* out_indices,
  cudf::size_type* d_counter);

template __global__ void iterate_join_map<true, int64_t>(
  mark_join_map_ref_type<int64_t> map_device_view,
  cudf::size_type* out_indices,
  cudf::size_type* d_counter);

template __global__ void iterate_join_map<false, int64_t>(
  mark_join_map_ref_type<int64_t> map_device_view,
  cudf::size_type* out_indices,
  cudf::size_type* d_counter);

}  // namespace q21
}  // namespace benchmark
}  // namespace gqe_python
