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

#pragma once

#include <utility/config.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include <cuda/std/optional>
#include <cuda/std/utility>

#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/detail/open_addressing/functors.cuh>  // slot_is_filled
#include <cuco/detail/open_addressing/open_addressing_ref_impl.cuh>
#include <cuco/operator.hpp>
#include <cuco/pair.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_multimap.cuh>
#include <cuco/types.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cstdint>

namespace {
// Shared memory must be declared outside of the `class`, either in the file
// scope or the function scope. See "D.3.1.6.3. Shared Memory Variable
// Declarations".
__shared__ cuda::atomic<cudf::size_type, cuda::thread_scope_block>
  cta_mark_counter;  /// The number of distinct hash map entries for which the
                     /// CTA set a mark bit.
}  // namespace

namespace gqe_python {
namespace utility {
/**
 * @brief Compute the mark mask for the data type of a hash map key.
 * @return The mask.
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
__host__ __device__ constexpr T mark_mask() noexcept
{
  std::size_t shift = sizeof(T) * std::size_t{8} - std::size_t{1};
  auto mask         = T{1} << shift;
  return mask;
}

/**
 * @brief Set the mark bit on a value.
 *
 * @pre The most significant bit of the value must be unused.
 *
 * @return The marked value.
 */
template <typename T>
__host__ __device__ constexpr T set_mark(T value) noexcept
{
  return value | mark_mask<T>();
}

/**
 * @brief Unset the mark bit on a value.
 *
 * @pre The most significant bit of the value must be unused.
 *
 * @return The unmarked value.
 */
template <typename T>
__host__ __device__ constexpr T unset_mark(T value) noexcept
{
  return value & ~mark_mask<T>();
}

/**
 * @brief Check if a  value is marked.
 *
 * @pre The most significant bit of the value must be unused.
 *
 * @return True if the value is marked, else false.
 */
template <typename T>
__host__ __device__ constexpr bool is_marked(T value) noexcept
{
  return value & mark_mask<T>();
}

/**
 * @brief Packed atomic load of a value.
 * @arg[in] address The memory address of the value.
 * @arg[in] memory_order A CUDA memory order.
 * @return The loaded value.
 *
 * @details The value of type `T` is loaded with a hardware-native atomic load
 * instruction.
 */
template <typename T>
__device__ T atomic_load(const T* address, cuda::memory_order memory_order)
{
  if constexpr (sizeof(T) <= 8) {
    using packed_type = cuda::std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    auto* slot_ptr    = reinterpret_cast<const packed_type*>(address);
    auto slot_ref     = cuda::atomic_ref<const packed_type, cuda::thread_scope_device>{*slot_ptr};

    auto slot_value = slot_ref.load(memory_order);
    return *reinterpret_cast<T*>(&slot_value);
  } else {
    auto first_ref =
      cuda::atomic_ref<const typename T::first_type, cuda::thread_scope_device>{address->first};
    auto second_ref =
      cuda::atomic_ref<const typename T::second_type, cuda::thread_scope_device>{address->second};

    return {first_ref.load(memory_order), second_ref.load(memory_order)};
  }
}

/**
 * @brief A {semi, anti} left mark join.
 * @pre The most significant bit of the hash map key must be unused.
 *
 * The mark join algorithm performs a semi-join (or anti-join). It builds the
 * hash map on the left hand side input table, and probes using the right hand
 * side input table. After the probe is complete, the join result can be
 * retrieved from the hash map.
 *
 * During the probe, the join marks entries as "seen". The retrieve uses the
 * mark to determine which entries belong to the result set. The result set
 * contains only distinct keys.
 */
template <typename hash_map_ref_type>
class mark_join_op {
  static constexpr bool has_payload = true;

  using slot_is_filled_type =
    cuco::detail::open_addressing_ns::slot_is_filled<has_payload,
                                                     typename hash_map_ref_type::key_type>;
  // Bucket size is used in cuco and cannot be set as template args to pass.
  // Each bucket contains `bucket_size` elements.
  static constexpr auto bucket_size = hash_map_ref_type::storage_ref_type::bucket_size;

 public:
  /**
     @brief A new mark join operation.
   */
  __device__ mark_join_op(
    hash_map_ref_type map_ref,
    cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_mark_counter_ref) noexcept
    : map_ref(map_ref),
      is_filled(slot_is_filled_type(map_ref.empty_key_sentinel(), map_ref.erased_key_sentinel())),
      predicate({map_ref.empty_key_sentinel(), map_ref.erased_key_sentinel(), map_ref.key_eq()}),
      mark_counter(0),
      global_mark_counter(global_mark_counter_ref)
  {
  }

  /**
   * @brief Prepare to `set_mark()`.
   * @pre Warp-synchronous function; must be called by all threads.
   * @pre Must be called before the probe loop.
   */
  __device__ void pre_set_mark_loop()
  {
    mark_counter = 0;
    if (threadIdx.x == leader) { cta_mark_counter.store(0, cuda::memory_order_relaxed); }
    __syncthreads();
  }

  /**
   * @brief Probe the hash map to set the mark bit.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   *
   * @note `emit_result_size = true` computes the exact join result size. This
   * is intended to allocate memory for `retrieve()`. However, disabled by
   * default for a slight speedup.
   */
  template <bool emit_result_size = false, typename probe_key_type, typename join_predicate_type>
  __device__ void set_mark(probe_key_type const* key_ptr,
                           join_predicate_type&& join_predicate,
                           bool is_active) noexcept
  {
    auto probing_scheme         = map_ref.probing_scheme();
    auto window_extent          = map_ref.bucket_extent();
    auto storage                = map_ref.storage_ref();
    auto empty_key_sentinel_key = map_ref.empty_key_sentinel();

    if (is_active) {
      // Perform read.
      probe_key_type key = *key_ptr;

      auto probing_iter = probing_scheme(key, window_extent);
      bool running      = true;
      while (true) {
        auto bucket_slots = (storage.data() + *probing_iter)->data();
#pragma unroll bucket_size
        for (int32_t i = 0; i < bucket_size; i++) {
          auto mutable_entry = bucket_slots + i;
          auto entry_value =
            gqe_python::utility::atomic_load(mutable_entry, cuda::memory_order_relaxed);
          auto cleaned_entry =
            cuco::make_pair(gqe_python::utility::unset_mark(entry_value.first), entry_value.second);

          auto status = predicate.operator()<cuco::detail::is_insert::NO>(key, cleaned_entry.first);

          if (status == cuco::detail::equal_result::EQUAL && join_predicate(cleaned_entry)) {
            auto expected = key;
            auto desired  = gqe_python::utility::set_mark(expected);

            cuda::atomic_ref<probe_key_type, cuda::thread_scope_device> atomic_entry{
              mutable_entry->first};

            if constexpr (emit_result_size) {
              bool is_success =
                atomic_entry.compare_exchange_strong(expected, desired, cuda::memory_order_relaxed);
              if (is_success) {
                // The marked entries count is the join result size. Thus,
                // don't double-count marked entries.
                ++mark_counter;
              }
            } else {
              if (!gqe_python::utility::is_marked(entry_value.first)) {
                atomic_entry.store(desired, cuda::memory_order_relaxed);
              }
            }
            // If key is unique, setting running = false would improve performance because of early
            // exit, otherwise, there might be accuracy issues.

            // running = false;
            // break;
          } else if (status == cuco::detail::equal_result::EMPTY) {
            running = false;
            break;
          }
        }
        if (!running) { break; }
        ++probing_iter;
      }
    }
  }

  /**
   * @brief Finalize to `set_mark()`.
   * @pre Warp-synchronous function; must be called by all threads.
   * @pre Must be called after the probe loop.
   */
  template <bool emit_result_size = false>
  __device__ void post_set_mark_loop() noexcept
  {
    if constexpr (emit_result_size) {
      cta_mark_counter.fetch_add(mark_counter, cuda::memory_order_relaxed);
      __syncthreads();

      if (threadIdx.x == leader) {
        global_mark_counter.fetch_add(cta_mark_counter.load(cuda::memory_order_relaxed),
                                      cuda::memory_order_relaxed);
      }
    } else {
      // Synchronize to avoid causing different behavior than the if-branch.
      __syncthreads();
    }
  }

  /**
   * @brief Retrieve the join result.
   * @pre CTA-synchronous function; must be called by all threads.
   * @invariant The callback is guaraneed to be called in CTA-synchronous
   * fashion. I.e., the callback is allowed to use, e.g., `__warpsync()` and
   * `__syncthreads()`.
   * @arg[in] callback_op The callback will be called for each join result row.
   * If the thread has a row, the will be passed to the callback, else
   * `nullopt`. The row type is
   * `cuda::std::optional<cuda::std::tuple<value_type>>`.
   */
  template <bool is_anti_join, typename callback_op_type>
  __device__ void retrieve(callback_op_type&& callback_op) noexcept
  {
    auto storage_ref          = map_ref.storage_ref();
    const auto loop_stride    = blockDim.x * gridDim.x;
    const size_t map_capacity = map_ref.capacity();
    const auto loop_bound =
      gqe::utility::divide_round_up((map_capacity + bucket_size - 1) / bucket_size, blockDim.x) *
      blockDim.x;
    cuda::std::optional<typename hash_map_ref_type::value_type::second_type> slot = {};
    for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < loop_bound;
         index += loop_stride) {
#pragma unroll bucket_size
      for (int32_t i = 0; i < bucket_size; i++) {
        slot.reset();
        if (index * bucket_size + i < map_capacity) {
          auto tmp = storage_ref[index][i];
          if (is_filled(tmp) && (gqe_python::utility::is_marked(tmp.first) ^ is_anti_join)) {
            slot = cuda::std::move(tmp.second);
          }
        }
        // Must be called by all threads.
        callback_op(slot);
      }
    }
  }

 private:
  static constexpr unsigned int leader = gqe_python::utility::warp_leader;
  static constexpr uint32_t full_mask  = gqe_python::utility::warp_full_mask;

  hash_map_ref_type map_ref;
  slot_is_filled_type is_filled;
  cuco::detail::equal_wrapper<typename hash_map_ref_type::key_type,
                              typename hash_map_ref_type::key_equal>
    predicate;

  cudf::size_type mark_counter;
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_mark_counter;
};

}  // namespace utility
}  // namespace gqe_python