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

#include <tpch/q22/fused_project_filter.cuh>

namespace gqe_python {
namespace benchmark {
namespace q22 {

// Q22 c_phone filter predicates.
__constant__ char c_phone_filters[7][3] = {"13", "31", "23", "29", "30", "18", "17"};

template <typename identifier_type, typename decimal_type>
__global__ void fused_project_filter_kernel(cudf::column_device_view c_phone_column,
                                            cudf::column_device_view c_acctbal_column,
                                            cudf::column_device_view c_custkey_column,
                                            cudf::size_type* d_counter,
                                            char* out_c_phone_column,
                                            cudf::mutable_column_device_view out_c_acctbal_column,
                                            cudf::mutable_column_device_view out_c_custkey_column)
{
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> d_global_offset(*d_counter);

  auto const row_count   = c_phone_column.size();
  auto const loop_stride = gridDim.x * blockDim.x;
  auto const loop_begin  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const loop_end = gqe::utility::divide_round_up(row_count, gqe_python::utility::warp_size) *
                        gqe_python::utility::warp_size;
  auto const lane_id = threadIdx.x % gqe_python::utility::warp_size;

  for (cudf::size_type idx = loop_begin; idx < loop_end; idx += loop_stride) {
    __syncwarp();
    bool is_active = idx < row_count;
    cudf::string_view phone_sv;

    if (is_active) {
      is_active = false;
      phone_sv  = c_phone_column.element<cudf::string_view>(idx);
      char a = phone_sv.data()[0], b = phone_sv.data()[1];
      for (int i = 0; i < 7; i++) {
        if (a == c_phone_filters[i][0] && b == c_phone_filters[i][1]) {
          is_active = true;
          break;
        }
      }
    }

    decimal_type c_acctbal    = c_acctbal_column.element<decimal_type>(idx);
    identifier_type c_custkey = c_custkey_column.element<identifier_type>(idx);

    if (is_active) { is_active = c_acctbal > 0.0; }

    unsigned mask    = __ballot_sync(gqe_python::utility::warp_full_mask, is_active);
    int num_to_write = __popc(mask);
    int warp_base    = 0;
    if (lane_id == 0 && num_to_write > 0) {
      warp_base = d_global_offset.fetch_add(num_to_write, cuda::memory_order_relaxed);
    }
    warp_base       = __shfl_sync(gqe_python::utility::warp_full_mask, warp_base, 0);
    int lane_prefix = __popc(mask & ((1u << lane_id) - 1));
    if (is_active) {
      int offset                                            = warp_base + lane_prefix;
      out_c_phone_column[2 * offset]                        = phone_sv.data()[0];
      out_c_phone_column[2 * offset + 1]                    = phone_sv.data()[1];
      out_c_acctbal_column.element<decimal_type>(offset)    = c_acctbal;
      out_c_custkey_column.element<identifier_type>(offset) = c_custkey;
    }
  }
}

std::unique_ptr<cudf::column> make_customized_string_column(rmm::device_uvector<char> d_chars,
                                                            cudf::size_type num)
{
  auto s = cudf::get_default_stream();

  rmm::device_uvector<cudf::size_type> d_offsets(num + 1, s);
  thrust::transform(rmm::exec_policy(s),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num + 1),
                    d_offsets.begin(),
                    [] __device__(cudf::size_type i) { return 2 * i; });

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    static_cast<cudf::size_type>(num + 1),
                                                    d_offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);

  return cudf::make_strings_column(
    num, std::move(offsets_col), d_chars.release(), 0, rmm::device_buffer{});
}

// Explicit instantiation of the fused_project_filter_kernel for the supported types of identifier
// type and decimal type.
template __global__ void fused_project_filter_kernel<int64_t, float>(
  cudf::column_device_view,
  cudf::column_device_view,
  cudf::column_device_view,
  cudf::size_type*,
  char*,
  cudf::mutable_column_device_view,
  cudf::mutable_column_device_view);

template __global__ void fused_project_filter_kernel<int64_t, double>(
  cudf::column_device_view,
  cudf::column_device_view,
  cudf::column_device_view,
  cudf::size_type*,
  char*,
  cudf::mutable_column_device_view,
  cudf::mutable_column_device_view);

template __global__ void fused_project_filter_kernel<int32_t, float>(
  cudf::column_device_view,
  cudf::column_device_view,
  cudf::column_device_view,
  cudf::size_type*,
  char*,
  cudf::mutable_column_device_view,
  cudf::mutable_column_device_view);

template __global__ void fused_project_filter_kernel<int32_t, double>(
  cudf::column_device_view,
  cudf::column_device_view,
  cudf::column_device_view,
  cudf::size_type*,
  char*,
  cudf::mutable_column_device_view,
  cudf::mutable_column_device_view);

}  // namespace q22
}  // namespace benchmark
}  // namespace gqe_python
