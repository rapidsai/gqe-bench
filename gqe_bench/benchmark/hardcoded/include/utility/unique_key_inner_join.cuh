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

#include <utility/config.hpp>

#include <cuda/std/optional>
#include <cuda/std/utility>

#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/types.cuh>

#include <gqe/utility/helpers.hpp>

#include <cstdint>

namespace gqe_python {
namespace utility {

/**
 * @brief A unique-key inner join.
 *
 * @pre The behavior is undefined if the key column on the build side contains
 * duplicate elements.
 *
 * The unique-key inner join algorithm builds a hash map of one of the input
 * tables (`build side`) on a key column assumed to contain unique values. The
 * key column in the other input table (`probe side`) is then used to search
 * for matching key-value pairs in the hash map, where the value indicates the
 * index of the row containing the found key in the build side table. Having
 * obtained the row index in the build side for each row of the probe that is
 * included in the join result, the result is constructed by projecting on the
 * desired output columns.
 *
 * Currently only supports singular key column.
 */
template <typename hash_map_ref_type>
class unique_key_inner_join_op {
  // Bucket size is used in cuco and cannot be set as template args to pass.
  // Each bucket contains `bucket_size` elements.
  static constexpr auto bucket_size = hash_map_ref_type::storage_ref_type::bucket_size;

 public:
  /**
     @brief A new unique-key inner join operation.
   */
  __device__ unique_key_inner_join_op(hash_map_ref_type map_ref) noexcept
    : map_ref(map_ref),
      predicate({map_ref.empty_key_sentinel(), map_ref.erased_key_sentinel(), map_ref.key_eq()})
  {
  }

  /**
   * @brief Probe the hash map to find the row index in the build side table
   * that matches the probe key. Then invoke `callback_op` on the build and
   * probe side row indices.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   */
  template <typename probe_key_type, typename join_predicate_type, typename callback_op_type>
  __device__ void probe(probe_key_type const* key_ptr,
                        cudf::size_type const probe_side_row_idx,
                        join_predicate_type&& join_predicate,
                        bool is_active,
                        callback_op_type&& callback_op) noexcept
  {
    auto probing_scheme         = map_ref.probing_scheme();
    auto window_extent          = map_ref.bucket_extent();
    auto storage                = map_ref.storage_ref();
    auto empty_key_sentinel_key = map_ref.empty_key_sentinel();

    cuda::std::optional<cuda::std::pair<cudf::size_type, cudf::size_type>> slot;

    if (is_active) {
      // Perform read.
      probe_key_type key = *key_ptr;

      auto probing_iter = probing_scheme(key, window_extent);
      bool running      = true;
      while (true) {
        auto bucket_slots = (storage.data() + *probing_iter)->data();
#pragma unroll bucket_size
        for (int32_t i = 0; i < bucket_size; i++) {
          auto const entry_value = *(bucket_slots + i);

          auto status = predicate.operator()<cuco::detail::is_insert::NO>(key, entry_value.first);

          if (status == cuco::detail::equal_result::EQUAL && join_predicate(entry_value)) {
            auto build_side_row_idx = entry_value.second;
            slot                    = cuda::std::make_pair(build_side_row_idx, probe_side_row_idx);

            // build-side keys are unique, thus exit probe loop early.
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
    // Must be called by all threads.
    callback_op(slot);
  }

 private:
  hash_map_ref_type map_ref;
  cuco::detail::equal_wrapper<typename hash_map_ref_type::key_type,
                              typename hash_map_ref_type::key_equal>
    predicate;
};

}  // namespace utility
}  // namespace gqe_python
