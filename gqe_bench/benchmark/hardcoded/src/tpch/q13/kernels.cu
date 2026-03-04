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

#include <tpch/q13/kernels.cuh>

#include <utility/config.hpp>
#include <utility/groupjoin.cuh>
#include <utility/write_buffer.cuh>

#include <cuda/atomic>
#include <cuda/std/utility>

#include <gqe/utility/helpers.hpp>

#include <cstdint>

namespace gqe_python {
namespace benchmark {
namespace q13 {

namespace {

constexpr int WORD1_LEN      = 7;
constexpr int WORD2_LEN      = 8;
constexpr int TOTAL_WORD_LEN = 15;

// Hard coded mask mappings:
//
// S P E C I A L
// S - 0b00000001 P - 0b00000010 E - 0b00000100 C - 0b00001000 I - 0b00010000 A - 0b00100000 L -
// 0b01000000 R E Q U E S T S R - 0b00000001 E - 0b00010010 Q - 0b00000100 U - 0b00001000 S -
// 0b10100000 T - 0b01000000
__device__ uint8_t mask_pattern_1[256] = {
  ['s'] = 0b00000001,
  ['p'] = 0b00000010,
  ['e'] = 0b00000100,
  ['c'] = 0b00001000,
  ['i'] = 0b00010000,
  ['a'] = 0b00100000,
  ['l'] = 0b01000000,
};

__device__ uint8_t mask_pattern_2[256] = {
  ['r'] = 0b00000001,
  ['e'] = 0b00010010,
  ['q'] = 0b00000100,
  ['u'] = 0b00001000,
  ['s'] = 0b10100000,
  ['t'] = 0b01000000,
};

// Shift-And algorithm applied to "%WORD1%WORD2%" pattern matching
inline __device__ bool check_if_comment_valid(cudf::string_view o_comment)
{
  bool valid_comment = true;

  bool found_w1        = false;
  uint64_t state       = 0;
  const uint8_t* mask  = mask_pattern_1;
  int word_len         = WORD1_LEN;
  uint64_t final_state = 1 << (word_len - 1);
  int farthest_start =
    o_comment.length() -
    TOTAL_WORD_LEN;  // If we have no WORD 1 overlap and are at this position, we can exit early

  if (o_comment.length() >= TOTAL_WORD_LEN) {
    for (int i = 0; i < o_comment.length(); i++) {
      state  = (state << 1) + 1;
      char c = o_comment[i];
      state  = state & mask[c];
      if ((state & final_state) !=
          0) {  // Matched WORD 1 so switching state and masks to check for WORD 2
        if (found_w1) {
          valid_comment = false;
          break;
        }
        // Switch to probing WORD2
        state       = 0;
        found_w1    = true;
        word_len    = WORD2_LEN;
        final_state = 1 << (word_len - 1);  // reset for WORD2
        mask        = mask_pattern_2;
        farthest_start =
          o_comment.length() -
          word_len;  // If we have no WORD 2 overlap and are at this position, we can exit early
      }
      if (i > farthest_start && (state == 0))
        break;  // Exit early if not possible to complete pattern in remaining characters
    }
  }

  return valid_comment;
}
}  // namespace

template <typename identifier_type>
__global__ void filter_orders_kernel(cudf::column_device_view o_custkey,
                                     cudf::column_device_view o_comment,
                                     cudf::size_type* d_global_offset,
                                     cudf::mutable_column_device_view out_o_custkey)
{
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_offset_ref(*d_global_offset);

  __shared__ typename utility::write_buffer_op<identifier_type>::storage_t wbs;

  utility::write_buffer_op<identifier_type> wb(
    &wbs, global_offset_ref, out_o_custkey.data<identifier_type>());

  auto const row_count   = o_custkey.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const loop_end =
    gqe::utility::divide_round_up(row_count, utility::warp_size) * utility::warp_size;

  for (cudf::size_type idx = loop_begin; idx < loop_end; idx += loop_stride) {
    __syncwarp();
    bool is_active = idx < row_count;

    if (is_active) {
      auto current_comment = o_comment.element<cudf::string_view>(idx);
      is_active            = check_if_comment_valid(current_comment);
    }

    // Late materialize the keys.
    cuda::std::optional<cuda::std::tuple<identifier_type>> tuple = cuda::std::nullopt;
    if (is_active) {
      auto custkey = o_custkey.element<identifier_type>(idx);
      tuple        = cuda::std::make_optional(cuda::std::make_tuple(custkey));
    }

    wb.write(tuple);
  }

  wb.flush();
}

template <typename identifier_type>
__global__ void groupjoin_probe_kernel(
  utility::map_ref_type<identifier_type, cudf::size_type> customer_map,
  cudf::column_device_view o_custkey)
{
  auto const row_count   = o_custkey.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const loop_end =
    gqe::utility::divide_round_up(row_count, utility::warp_size) * utility::warp_size;

  utility::groupjoin_op gjo(customer_map);

  for (cudf::size_type idx = loop_begin; idx < loop_end; idx += loop_stride) {
    __syncwarp();
    bool is_active = idx < row_count;

    auto key            = o_custkey.element<identifier_type>(idx);
    auto join_predicate = [&](auto const& entry) -> bool { return key == entry.first; };

    gjo.probe(&key, join_predicate, is_active);
  }
}

template <typename identifier_type>
__global__ void fused_filter_probe_kernel(
  utility::map_ref_type<identifier_type, cudf::size_type> customer_map,
  cudf::column_device_view o_custkey,
  cudf::column_device_view o_comment)
{
  auto const row_count   = o_custkey.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const loop_end =
    gqe::utility::divide_round_up(row_count, utility::warp_size) * utility::warp_size;

  utility::groupjoin_op gjo(customer_map);

  for (cudf::size_type idx = loop_begin; idx < loop_end; idx += loop_stride) {
    __syncwarp();
    bool is_active = idx < row_count;

    if (is_active) {
      auto current_comment = o_comment.element<cudf::string_view>(idx);
      is_active            = check_if_comment_valid(current_comment);
    }

    // Avoid warp divergence, because it kills hash map lookup performance.
    __syncwarp();

    // Late materialize the key.
    auto key            = is_active ? o_custkey.element<identifier_type>(idx) : 0;
    auto join_predicate = [&](auto const& entry) -> bool { return key == entry.first; };

    gjo.probe(&key, join_predicate, is_active);
  }
}

template <typename identifier_type>
__global__ void groupjoin_retrieve_kernel(
  utility::map_ref_type<identifier_type, cudf::size_type> customer_map,
  cudf::size_type* d_global_offset,
  cudf::mutable_column_device_view out_c_custkey,
  cudf::mutable_column_device_view out_c_count)
{
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_offset_ref(*d_global_offset);

  __shared__ typename utility::write_buffer_op<identifier_type, cudf::size_type>::storage_t wbs;

  // Handle alignment of 64-bit identifiers by setting c_custkey as first
  // column, and 32-bit c_count as second column. That means we align on the
  // larger of the two types.
  utility::write_buffer_op<identifier_type, cudf::size_type> wb(
    &wbs,
    global_offset_ref,
    out_c_custkey.data<identifier_type>(),
    out_c_count.data<cudf::size_type>());

  utility::groupjoin_op gjo(customer_map);

  gjo.template retrieve([&wb](auto const& slot) { wb.write(slot); });

  wb.flush();
}

template __global__ void filter_orders_kernel<int32_t>(
  cudf::column_device_view o_custkey,
  cudf::column_device_view o_comment,
  cudf::size_type* d_global_offset,
  cudf::mutable_column_device_view out_o_custkey);

template __global__ void filter_orders_kernel<int64_t>(
  cudf::column_device_view o_custkey,
  cudf::column_device_view o_comment,
  cudf::size_type* d_global_offset,
  cudf::mutable_column_device_view out_o_custkey);

template __global__ void groupjoin_probe_kernel<int32_t>(
  utility::map_ref_type<int32_t, cudf::size_type> customer_map, cudf::column_device_view o_custkey);

template __global__ void groupjoin_probe_kernel<int64_t>(
  utility::map_ref_type<int64_t, cudf::size_type> customer_map, cudf::column_device_view o_custkey);

template __global__ void fused_filter_probe_kernel<int32_t>(
  utility::map_ref_type<int32_t, cudf::size_type> customer_map,
  cudf::column_device_view o_custkey,
  cudf::column_device_view o_comment);

template __global__ void fused_filter_probe_kernel<int64_t>(
  utility::map_ref_type<int64_t, cudf::size_type> customer_map,
  cudf::column_device_view o_custkey,
  cudf::column_device_view o_comment);

template __global__ void groupjoin_retrieve_kernel<int32_t>(
  utility::map_ref_type<int32_t, cudf::size_type> customer_map,
  cudf::size_type* d_global_offset,
  cudf::mutable_column_device_view out_c_custkey,
  cudf::mutable_column_device_view out_c_count);

template __global__ void groupjoin_retrieve_kernel<int64_t>(
  utility::map_ref_type<int64_t, cudf::size_type> customer_map,
  cudf::size_type* d_global_offset,
  cudf::mutable_column_device_view out_c_custkey,
  cudf::mutable_column_device_view out_c_count);

}  // namespace q13
}  // namespace benchmark
}  // namespace gqe_python
