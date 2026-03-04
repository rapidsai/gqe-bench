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

#include <tpch/q16/fused_filter_join.cuh>

namespace gqe_python {
namespace benchmark {
namespace q16 {

/**
 * @brief
 * The filter predicate is:
 * p_brand <> 'Brand#45'
 * and p_type not like 'MEDIUM POLISHED%'
 * and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
 */

__constant__ int32_t filter_p_size_list[] = {49, 14, 23, 45, 19, 3, 36, 9};
const size_t filter_p_size_list_length    = 8;
__constant__ char filter_p_brand[]        = "Brand#45";
const size_t filter_p_brand_length        = 8;
__constant__ char filter_p_type[]         = "MEDIUM POLISHED";
const size_t filter_p_type_length         = 15;

__device__ bool evaluate_part_filter(int32_t p_size,
                                     cudf::string_view p_brand,
                                     cudf::string_view p_type)
{
  // Size filter: p_size in (49, 14, 23, 45, 19, 3, 36, 9)
  bool is_active = false;
  for (int i = 0; i < filter_p_size_list_length; i++) {
    if (p_size == filter_p_size_list[i]) {
      is_active = true;
      break;
    }
  }

  // Brand filter: p_brand <> 'Brand#45'
  if (is_active) {
    bool match = true;
    for (int i = 0; i < filter_p_brand_length; i++) {
      if (p_brand.data()[i] != filter_p_brand[i]) {
        match = false;
        break;
      }
    }
    is_active = !match;
  }

  // Type filter: p_type not like 'MEDIUM POLISHED%'
  if (is_active) {
    bool should_exclude = false;
    if (p_type.length() >= filter_p_type_length) {
      should_exclude = true;
      for (int i = 0; i < filter_p_type_length; i++) {
        if (p_type.data()[i] != filter_p_type[i]) {
          should_exclude = false;
          break;
        }
      }
    }
    is_active = !should_exclude;
  }

  return is_active;
}

template <typename Identifier, bool EnableBloomFilter>
__global__ void fused_filter_hashtable_build_kernel(
  cudf::column_device_view p_partkey_column,
  cudf::column_device_view p_brand_column,
  cudf::column_device_view p_type_column,
  cudf::column_device_view p_size_column,
  part_map_ref_type<Identifier> part_map,
  gqe_python::utility::bloom_filter_ref_type<Identifier> bloom_filter)
{
  // Build the hash table for the part map.
  auto const row_count   = p_partkey_column.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto tile = cooperative_groups::tiled_partition<1>(cooperative_groups::this_thread_block());
  for (std::size_t idx = loop_begin; idx < row_count; idx += loop_stride) {
    auto const p_size  = p_size_column.element<int32_t>(idx);
    auto const p_brand = p_brand_column.element<cudf::string_view>(idx);
    auto const p_type  = p_type_column.element<cudf::string_view>(idx);
    bool is_active     = evaluate_part_filter(p_size, p_brand, p_type);
    if (is_active) {
      auto const p_partkey = p_partkey_column.element<Identifier>(idx);
      auto kv              = thrust::make_pair(p_partkey, idx);
      part_map.insert(tile, kv);
      if constexpr (EnableBloomFilter) { bloom_filter.add(p_partkey); }
    }
  }
}

template <typename Identifier, bool EnablePartBloomFilter, bool EnableSupplierBloomFilter>
__global__ void fused_join_kernel(fused_join_params<Identifier> params)
{
  auto ps_partkey_column     = params.ps_partkey_column;
  auto ps_suppkey_column     = params.ps_suppkey_column;
  auto supplier_map          = params.supplier_map;
  auto supplier_bloom_filter = params.supplier_bloom_filter;
  auto part_map              = params.part_map;
  auto part_bloom_filter     = params.part_bloom_filter;
  auto p_out_indices         = params.p_out_indices;
  auto ps_out_indices        = params.ps_out_indices;
  auto d_counter             = params.d_counter;

  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> d_global_offset(*d_counter);

  auto const row_count   = ps_partkey_column.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const loop_end = gqe::utility::divide_round_up(row_count, gqe_python::utility::warp_size) *
                        gqe_python::utility::warp_size;
  __shared__ typename utility::write_buffer_op<cudf::size_type, cudf::size_type>::storage_t wbs;
  utility::write_buffer_op<cudf::size_type, cudf::size_type> wb(
    &wbs, d_global_offset, p_out_indices, ps_out_indices);
  cuda::std::optional<cuda::std::tuple<Identifier, Identifier>> slot;
  for (std::size_t idx = loop_begin; idx < loop_end; idx += loop_stride) {
    __syncwarp();
    slot.reset();
    bool is_active          = idx < row_count;
    Identifier p_out_indice = -1, ps_out_indice = -1;
    Identifier ps_partkey = is_active ? ps_partkey_column.template element<Identifier>(idx) : -1;
    // Inner join.
    __syncwarp();
    if constexpr (EnablePartBloomFilter) {
      if (is_active) { is_active = part_bloom_filter.contains(ps_partkey); }
    }
    if (is_active) {
      auto const part_tuple = part_map.find(ps_partkey);
      if (part_tuple != part_map.end()) {
        // The map contains unique key, don't bother iterating.
        p_out_indice = part_tuple->second;
      } else {
        is_active = false;
      }
    }
    // Left anti join
    if (is_active) {
      Identifier ps_suppkey = ps_suppkey_column.template element<Identifier>(idx);
      if constexpr (EnableSupplierBloomFilter) {
        // If bloom filter says no, then definitely no match.
        // If bloom filter says yes, then we need to check the hash map.
        // This is because bloom filter may have false positives.
        if (supplier_bloom_filter.contains(ps_suppkey)) {
          auto const supp_tuple = supplier_map.find(ps_suppkey);
          is_active             = (supp_tuple == supplier_map.end());
        }
      } else {
        auto const supp_tuple = supplier_map.find(ps_suppkey);
        is_active             = (supp_tuple == supplier_map.end());
      }
    }
    if (is_active) { ps_out_indice = idx; }
    // Write results into output buffers.
    if (is_active) { slot = {p_out_indice, ps_out_indice}; }
    wb.write(slot);
  }
  wb.flush();
}

template __global__ void fused_filter_hashtable_build_kernel<int32_t, true>(
  cudf::column_device_view p_partkey_column,
  cudf::column_device_view p_brand_column,
  cudf::column_device_view p_type_column,
  cudf::column_device_view p_size_column,
  part_map_ref_type<int32_t> part_map,
  gqe_python::utility::bloom_filter_ref_type<int32_t> bloom_filter);

template __global__ void fused_filter_hashtable_build_kernel<int32_t, false>(
  cudf::column_device_view p_partkey_column,
  cudf::column_device_view p_brand_column,
  cudf::column_device_view p_type_column,
  cudf::column_device_view p_size_column,
  part_map_ref_type<int32_t> part_map,
  gqe_python::utility::bloom_filter_ref_type<int32_t> bloom_filter);

template __global__ void fused_filter_hashtable_build_kernel<int64_t, true>(
  cudf::column_device_view p_partkey_column,
  cudf::column_device_view p_brand_column,
  cudf::column_device_view p_type_column,
  cudf::column_device_view p_size_column,
  part_map_ref_type<int64_t> part_map,
  gqe_python::utility::bloom_filter_ref_type<int64_t> bloom_filter);

template __global__ void fused_filter_hashtable_build_kernel<int64_t, false>(
  cudf::column_device_view p_partkey_column,
  cudf::column_device_view p_brand_column,
  cudf::column_device_view p_type_column,
  cudf::column_device_view p_size_column,
  part_map_ref_type<int64_t> part_map,
  gqe_python::utility::bloom_filter_ref_type<int64_t> bloom_filter);

template __global__ void fused_join_kernel<int32_t, true, true>(fused_join_params<int32_t> params);
template __global__ void fused_join_kernel<int32_t, true, false>(fused_join_params<int32_t> params);
template __global__ void fused_join_kernel<int32_t, false, true>(fused_join_params<int32_t> params);
template __global__ void fused_join_kernel<int32_t, false, false>(
  fused_join_params<int32_t> params);

template __global__ void fused_join_kernel<int64_t, true, true>(fused_join_params<int64_t> params);
template __global__ void fused_join_kernel<int64_t, true, false>(fused_join_params<int64_t> params);
template __global__ void fused_join_kernel<int64_t, false, true>(fused_join_params<int64_t> params);
template __global__ void fused_join_kernel<int64_t, false, false>(
  fused_join_params<int64_t> params);

}  // namespace q16
}  // namespace benchmark
}  // namespace gqe_python
