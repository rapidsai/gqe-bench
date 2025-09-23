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

#include <gqe/utility/helpers.hpp>

#include <cstdint>

namespace gqe_python {
namespace utility {

/**
 * @brief A groupjoin operator for TPC-H Q13.
 *
 * @pre The aggregate function is currently a `COUNT(*)`.
 *
 * The groupjoin algorithm performs a combination of a join and a group by,
 * whereby the join and grouping are on the same key. The purpose is to share
 * the hash map between the two operations, and avoid materializing the
 * intermediate result.
 *
 * A side-benefit is that, in contrast to a regular group by, the keys are
 * already inserted before the aggregation phase. Therefore, the aggregation can
 * directly aggregate and does not need to atomically initialize a hash map
 * entry.
 *
 * The algorithm works in three phases:
 *
 * 1. Build. Equivalent to the build of a join, but additionally sets the aggregate values to the
 * "identity element" (see https://en.wikipedia.org/wiki/Identity_element).
 * 2. Probe. Effectively a fused join probe and group by aggregation.
 *    Instead of returning a match, the probe executes the aggregate function on the
 *    matching entry.
 * 3. Retrieve. Retrieves the valid entries from the hash map and stores them as a
 *    cuDF table.
 *
 * The build is a pipeline blocker, because the hash map must be completed
 * before the probe. The probe is also a pipeline blocker, because the
 * aggregation must be completed before the retrieve.
 *
 * The build phase can be executed concurrently on different row groups, and
 * pipelined with its inputs. The same holds for the probe phase.
 *
 * # References
 *
 * Moerkotte and Neumann "Accelerating queries with group-by and join by groupjoin" PVLDB 2011
 */
template <typename hash_map_ref_type>
class groupjoin_op {
  static constexpr bool has_payload = true;

  using slot_is_filled_type =
    cuco::detail::open_addressing_ns::slot_is_filled<has_payload,
                                                     typename hash_map_ref_type::key_type>;
  // Bucket size is used in cuco and cannot be set as template args to pass.
  // Each bucket contains `bucket_size` elements.
  static constexpr auto bucket_size = hash_map_ref_type::storage_ref_type::bucket_size;

 public:
  /**
     @brief A new groupjoin operation.
   */
  __device__ groupjoin_op(hash_map_ref_type map_ref) noexcept
    : map_ref(map_ref),
      is_filled(slot_is_filled_type(map_ref.empty_key_sentinel(), map_ref.erased_key_sentinel())),
      predicate({map_ref.empty_key_sentinel(), map_ref.erased_key_sentinel(), map_ref.key_eq()})
  {
  }

  /**
   * @brief Probe the hash map and perform the aggregate function.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   *
   * @arg[in] key_ptr The groupjoin key. Dereferenced inside the probe to facilicate late
   * materialization.
   * @arg[in] join_predicate The join predicate.
   * @arg[in] is_active Flag indicating that the thread is active and thus the key_ptr is a valid
   * memory location.
   */
  template <typename probe_key_type, typename join_predicate_type>
  __device__ void probe(probe_key_type const* key_ptr,
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
          auto entry_value   = *mutable_entry;

          auto status = predicate.operator()<cuco::detail::is_insert::NO>(key, entry_value.first);

          if (status == cuco::detail::equal_result::EQUAL && join_predicate(entry_value)) {
            cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> atomic_entry{
              mutable_entry->second};

            atomic_entry.fetch_add(1);

            // Group by semantically uniques each key. I.e., there will never be
            // a duplicate key in the hash map. Thus, exit probe loop early.
            running = false;
            break;
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
   * @brief Retrieve the groupjoin result.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   *
   * @invariant The callback is guaranteed to be called in warp-synchronous
   * fashion. I.e., the callback is allowed to use, e.g., `__warpsync()`.
   *
   * @arg[in] callback_op The callback will be called for each join result row.
   * If the thread has a row, the row will be passed to the callback, else
   * `nullopt`. The row type is
   * `cuda::std::optional<cuda::std::tuple<value_type>>`.
   */
  template <typename callback_op_type>
  __device__ void retrieve(callback_op_type&& callback_op) noexcept
  {
    auto storage_ref          = map_ref.storage_ref();
    const auto loop_stride    = blockDim.x * gridDim.x;
    const size_t map_capacity = map_ref.capacity();
    const auto loop_bound =
      gqe::utility::divide_round_up((map_capacity + bucket_size - 1) / bucket_size, blockDim.x) *
      blockDim.x;

    cuda::std::optional<cuda::std::pair<typename hash_map_ref_type::value_type::first_type,
                                        typename hash_map_ref_type::value_type::second_type>>
      slot = {};

    for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < loop_bound;
         index += loop_stride) {
#pragma unroll bucket_size
      for (int32_t i = 0; i < bucket_size; i++) {
        slot.reset();
        if (index * bucket_size + i < map_capacity) {
          auto tmp = storage_ref[index][i];
          if (is_filled(tmp)) {
            slot = cuda::std::move(cuda::std::make_pair(tmp.first, tmp.second));
          }
        }
        // Must be called by all threads.
        callback_op(slot);
      }
    }
  }

 private:
  hash_map_ref_type map_ref;
  slot_is_filled_type is_filled;
  cuco::detail::equal_wrapper<typename hash_map_ref_type::key_type,
                              typename hash_map_ref_type::key_equal>
    predicate;
};

}  // namespace utility
}  // namespace gqe_python
