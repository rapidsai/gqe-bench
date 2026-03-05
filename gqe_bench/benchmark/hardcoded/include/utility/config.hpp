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

#include <cuco/bloom_filter.cuh>
#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>

// Specialization to make double work with cuco
namespace cuco {
template <>
struct is_bitwise_comparable<double> : std::true_type {};
}  // namespace cuco

#include <cuda/functional>
#include <cuda/std/cstddef>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cstdlib>

namespace gqe_python {
namespace utility {
constexpr int32_t cg_size            = 1;
constexpr int32_t block_dim          = 256;
constexpr std::size_t max_block_size = 1024;  /// The maximum number of threads in a thread block.
constexpr int32_t warp_size          = 32;
static constexpr std::size_t max_num_warps =
  max_block_size / warp_size;  /// The maximum number of warps in a thread block.
constexpr int32_t output_cache_size = 32;
constexpr uint32_t warp_full_mask   = 0xFFFFFFFFu;
constexpr int warp_leader           = 0;
constexpr int32_t bucket_size       = 2;

/**
 * @brief Stream-ordered allocator adaptor used for cuco data structures
 *
 * Copied from cudf
 * https://github.com/rapidsai/cudf/blob/branch-25.02/cpp/include/cudf/detail/cuco_helpers.hpp#L32-L43
 *
 * The stream-ordered `rmm::mr::polymorphic_allocator` cannot be used in `cuco`
 * directly since the later expects a standard C++ `Allocator` interface. This
 * allocator helper provides a simple way to handle cuco memory
 * allocation/deallocation with the given `stream` and the rmm default memory
 * resource.
 */
template <typename T>
using cuco_allocator = rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<T>>;
template <typename Key, typename T>
using map_allocator_type = cuco_allocator<cuco::pair<Key, T>>;
template <typename Key, typename T>
using map_allocator_instance_type          = rmm::mr::polymorphic_allocator<cuco::pair<Key, T>>;
using bloom_filter_allocator_type          = cuco_allocator<cuda::std::byte>;
using bloom_filter_allocator_instance_type = rmm::mr::polymorphic_allocator<cuda::std::byte>;

template <typename Key, typename T, typename Hash = cuco::default_hash_function<Key>>
using map_type = cuco::static_map<Key,
                                  T,
                                  cuco::extent<std::size_t>,
                                  cuda::thread_scope_device,
                                  thrust::equal_to<Key>,
                                  cuco::linear_probing<cg_size, Hash>,
                                  map_allocator_type<Key, T>,
                                  cuco::storage<bucket_size>>;
template <typename Key, typename T, typename Hash = cuco::default_hash_function<Key>>
using map_ref_type =
  typename map_type<Key, T, Hash>::template ref_type<cuco::op::find_tag, cuco::op::for_each_tag>;

template <typename Key, typename T>
using multimap_type = cuco::experimental::static_multimap<
  Key,
  T,
  cuco::extent<std::size_t>,
  cuda::thread_scope_device,
  thrust::equal_to<Key>,
  cuco::linear_probing<cg_size, cuco::default_hash_function<Key>>,
  map_allocator_type<Key, T>,
  cuco::storage<bucket_size>>;
template <typename Key, typename T>
using multimap_ref_type =
  typename multimap_type<Key, T>::template ref_type<cuco::op::find_tag, cuco::op::for_each_tag>;

// Q18-specific type aliases with bucket_size = 1
namespace q18 {
constexpr int32_t bucket_size = 1;

template <typename Key, typename T, typename Hash = cuco::default_hash_function<Key>>
using map_type = cuco::static_map<Key,
                                  T,
                                  cuco::extent<std::size_t>,
                                  cuda::thread_scope_device,
                                  thrust::equal_to<Key>,
                                  cuco::linear_probing<cg_size, Hash>,
                                  map_allocator_type<Key, T>,
                                  cuco::storage<bucket_size>>;  // Uses q18::bucket_size = 1
template <typename Key, typename T, typename Hash = cuco::default_hash_function<Key>>
using map_ref_type =
  typename map_type<Key, T, Hash>::template ref_type<cuco::op::find_tag, cuco::op::for_each_tag>;
}  // namespace q18

template <typename T, typename Hash = cuco::xxhash_64<T>>
using bloom_filter_policy_type = cuco::default_filter_policy<Hash, std::uint32_t, 2>;
template <typename T, typename Hash = cuco::xxhash_64<T>>
using bloom_filter_type = cuco::bloom_filter<T,
                                             cuco::extent<std::size_t>,
                                             cuda::thread_scope_device,
                                             bloom_filter_policy_type<T, Hash>,
                                             bloom_filter_allocator_type>;
template <typename T, typename Hash = cuco::xxhash_64<T>>
using bloom_filter_ref_type = typename bloom_filter_type<T, Hash>::template ref_type<>;

enum JOIN_TYPE { LEFT_SEMI_JOIN = 0, LEFT_ANTI_JOIN = 1 };
}  // namespace utility
}  // namespace gqe_python
