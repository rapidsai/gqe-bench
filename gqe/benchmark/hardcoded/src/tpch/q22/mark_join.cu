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

#include <tpch/q22/mark_join.cuh>

namespace gqe_python {
namespace benchmark {
namespace q22 {

template <typename identifier_type>
__global__ void mark_join_kernel(
  mark_join_map_ref_type<identifier_type> customer_map,
  gqe_python::utility::bloom_filter_ref_type<identifier_type> bloom_filter,
  cudf::column_device_view o_custkey_column,
  cudf::size_type* d_counter)
{
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> d_global_offset(*d_counter);
  gqe_python::utility::mark_join_op<mark_join_map_ref_type<identifier_type>> mjo(customer_map,
                                                                                 d_global_offset);
  mjo.pre_set_mark_loop();
  auto const row_count   = o_custkey_column.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const loop_end = gqe::utility::divide_round_up(row_count, gqe_python::utility::warp_size) *
                        gqe_python::utility::warp_size;

  for (cudf::size_type idx = loop_begin; idx < loop_end; idx += loop_stride) {
    __syncwarp();
    bool is_active            = idx < row_count;
    identifier_type o_custkey = is_active ? o_custkey_column.element<identifier_type>(idx) : -1;
    if (is_active) { is_active = bloom_filter.contains(o_custkey); }
    auto join_predicate = [&](auto const& entry) -> bool { return o_custkey == entry.first; };
    mjo.template set_mark<false>(&o_custkey, join_predicate, is_active);
  }

  mjo.template post_set_mark_loop<false>();
}

template <typename identifier_type>
__global__ void iterate_join_map(mark_join_map_ref_type<identifier_type> map_device_view,
                                 cudf::size_type* customer_out_indices,
                                 cudf::size_type* d_counter_ref)
{
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> d_global_offset(*d_counter_ref);
  constexpr bool is_anti_join = true;

  __shared__ typename utility::write_buffer_op<cudf::size_type>::storage_t wbs;
  utility::write_buffer_op<cudf::size_type> wb(&wbs, d_global_offset, customer_out_indices);

  gqe_python::utility::mark_join_op<mark_join_map_ref_type<identifier_type>> mjo(map_device_view,
                                                                                 d_global_offset);
  mjo.template retrieve<is_anti_join>([&wb](auto const& slot) { wb.write(slot); });

  wb.flush();
}

// Explicit instantiation of the mark_join_kernel for the supported types int32_t and int64_t.
template __global__ void mark_join_kernel<int32_t>(
  mark_join_map_ref_type<int32_t> customer_map,
  gqe_python::utility::bloom_filter_ref_type<int32_t> bloom_filter,
  cudf::column_device_view o_custkey_column,
  cudf::size_type* d_counter);

template __global__ void mark_join_kernel<int64_t>(
  mark_join_map_ref_type<int64_t> customer_map,
  gqe_python::utility::bloom_filter_ref_type<int64_t> bloom_filter,
  cudf::column_device_view o_custkey_column,
  cudf::size_type* d_counter);

template __global__ void iterate_join_map<int32_t>(mark_join_map_ref_type<int32_t> map_device_view,
                                                   cudf::size_type* customer_out_indices,
                                                   cudf::size_type* d_counter_ref);

template __global__ void iterate_join_map<int64_t>(mark_join_map_ref_type<int64_t> map_device_view,
                                                   cudf::size_type* customer_out_indices,
                                                   cudf::size_type* d_counter_ref);
}  // namespace q22
}  // namespace benchmark
}  // namespace gqe_python
