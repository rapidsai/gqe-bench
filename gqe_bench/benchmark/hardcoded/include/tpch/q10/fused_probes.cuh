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

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>

namespace gqe_python {
namespace benchmark {
namespace q10 {

/**
 * @brief A fused probes operation.
 *
 * First probe o_custkey_multimap with c_custkey and iterate over the matches
 * to also probe n_nationkey_map with c_nationkey. If a match is found on the
 * second probe as well, write the row indices into join_orders_lineitem_table,
 * nation_table, and customer_table to output.
 *
 */
template <typename hash_multimap_ref_type, typename hash_map_ref_type>
class fused_probes_op {
  // Bucket size is used in cuco and cannot be set as template args to pass.
  // Each bucket contains `bucket_size` elements.
  static constexpr auto o_custkey_multimap_bucket_size =
    hash_multimap_ref_type::storage_ref_type::bucket_size;
  static constexpr auto n_nationkey_map_bucket_size =
    hash_map_ref_type::storage_ref_type::bucket_size;

 public:
  /**
     @brief A new fused probes operation.
   */
  __device__ fused_probes_op(hash_multimap_ref_type o_custkey_multimap,
                             hash_map_ref_type n_nationkey_map) noexcept
    : o_custkey_multimap(o_custkey_multimap),
      n_nationkey_map(n_nationkey_map),
      o_custkey_multimap_predicate({o_custkey_multimap.empty_key_sentinel(),
                                    o_custkey_multimap.erased_key_sentinel(),
                                    o_custkey_multimap.key_eq()}),
      n_nationkey_map_predicate({n_nationkey_map.empty_key_sentinel(),
                                 n_nationkey_map.erased_key_sentinel(),
                                 n_nationkey_map.key_eq()})
  {
  }

  /**
   * @brief Probe n_nationkey_map with c_nationkey, saving results in slot.
   */
  template <typename probe_key_type>
  __device__ void probe_n_nationkey_map(
    probe_key_type const c_nationkey,
    cudf::size_type const join_orders_lineitem_table_row_idx,
    cudf::size_type const customer_table_row_idx,
    cuda::std::optional<cuda::std::array<cudf::size_type, 3>>& slot) noexcept
  {
    auto n_nationkey_map_probing_scheme         = n_nationkey_map.probing_scheme();
    auto n_nationkey_map_window_extent          = n_nationkey_map.bucket_extent();
    auto n_nationkey_map_storage                = n_nationkey_map.storage_ref();
    auto n_nationkey_map_empty_key_sentinel_key = n_nationkey_map.empty_key_sentinel();

    auto n_nationkey_map_probing_iter =
      n_nationkey_map_probing_scheme(c_nationkey, n_nationkey_map_window_extent);
    bool running = true;
    while (true) {
      auto n_nationkey_map_bucket_slots =
        (n_nationkey_map_storage.data() + *n_nationkey_map_probing_iter)->data();
#pragma unroll n_nationkey_map_bucket_size
      for (int32_t i = 0; i < n_nationkey_map_bucket_size; i++) {
        auto const n_nationkey_map_entry_value = *(n_nationkey_map_bucket_slots + i);

        auto status = n_nationkey_map_predicate.operator()<cuco::detail::is_insert::NO>(
          c_nationkey, n_nationkey_map_entry_value.first);
        auto nationkey_join_predicate = [&](auto const& entry) -> bool {
          return c_nationkey == entry.first;
        };

        if (status == cuco::detail::equal_result::EQUAL &&
            nationkey_join_predicate(n_nationkey_map_entry_value)) {
          auto nation_table_row_idx = n_nationkey_map_entry_value.second;
          slot = {join_orders_lineitem_table_row_idx, nation_table_row_idx, customer_table_row_idx};

          // n_nationkey_map keys are unique, thus exit probe loop early.
          running = false;
          break;
        } else if (status == cuco::detail::equal_result::EMPTY) {
          running = false;
          break;
        }
      }
      if (!running) { break; }
      ++n_nationkey_map_probing_iter;
    }
  }

  /**
   * @brief Probe o_custkey_multimap with c_custkey then n_nationkey_map with
   * c_nationkey. Then invoke `callback_op` on each tuple of output row
   * indices.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   */
  template <typename probe_key_type, typename callback_op_type>
  __device__ void fused_probes(probe_key_type const c_custkey,
                               cudf::size_type const customer_table_row_idx,
                               cudf::column_device_view c_nationkey_column,
                               bool is_active,
                               callback_op_type&& callback_op) noexcept
  {
    auto o_custkey_multimap_probing_scheme         = o_custkey_multimap.probing_scheme();
    auto o_custkey_multimap_window_extent          = o_custkey_multimap.bucket_extent();
    auto o_custkey_multimap_storage                = o_custkey_multimap.storage_ref();
    auto o_custkey_multimap_empty_key_sentinel_key = o_custkey_multimap.empty_key_sentinel();

    cuda::std::optional<cuda::std::array<cudf::size_type, 3>> slot;

    // Perform read.
    auto o_custkey_multimap_probing_iter =
      is_active
        ? o_custkey_multimap_probing_scheme(c_custkey, o_custkey_multimap_window_extent)
        : o_custkey_multimap_probing_scheme(probe_key_type{}, o_custkey_multimap_window_extent);
    while (true) {
      auto o_custkey_multimap_bucket_slots =
        is_active ? (o_custkey_multimap_storage.data() + *o_custkey_multimap_probing_iter)->data()
                  : nullptr;
#pragma unroll o_custkey_multimap_bucket_size
      for (int32_t i = 0; i < o_custkey_multimap_bucket_size; i++) {
        slot.reset();
        if (is_active) {
          auto const o_custkey_multimap_entry_value = *(o_custkey_multimap_bucket_slots + i);

          auto status = o_custkey_multimap_predicate.operator()<cuco::detail::is_insert::NO>(
            c_custkey, o_custkey_multimap_entry_value.first);
          auto custkey_join_predicate = [&](auto const& entry) -> bool {
            return c_custkey == entry.first;
          };
          if (status == cuco::detail::equal_result::EQUAL &&
              custkey_join_predicate(o_custkey_multimap_entry_value)) {
            auto join_orders_lineitem_table_row_idx = o_custkey_multimap_entry_value.second;
            auto c_nationkey = c_nationkey_column.element<probe_key_type>(customer_table_row_idx);
            // conditionally populate slot
            probe_n_nationkey_map(
              c_nationkey, join_orders_lineitem_table_row_idx, customer_table_row_idx, slot);
            // cannot exit probe loop early as there can be multiple matches
          } else if (status == cuco::detail::equal_result::EMPTY) {
            is_active = false;
          }
        }
        // Must be called by all threads.
        callback_op(slot);
      }
      if (__any_sync(0xFFFFFFFF, is_active)) {
        if (is_active) ++o_custkey_multimap_probing_iter;
      } else {
        break;
      }
    }
  }

 private:
  hash_multimap_ref_type o_custkey_multimap;
  hash_map_ref_type n_nationkey_map;
  cuco::detail::equal_wrapper<typename hash_multimap_ref_type::key_type,
                              typename hash_multimap_ref_type::key_equal>
    o_custkey_multimap_predicate;
  cuco::detail::equal_wrapper<typename hash_map_ref_type::key_type,
                              typename hash_map_ref_type::key_equal>
    n_nationkey_map_predicate;
};

// struct for passing output columns to fused_probes_kernel
// This is a workaround to a compiler issue where heavily templated data structures result in long
// symbol names that lead to mismatched function declaration/definition in cudafe-generated code. In
// `fused_probes_kernel`, the parameters with long string representations are the hash (multi)maps
// and the Bloom filter, but for some reason this particular issue only manifests when multiple
// parameters of the type `cudf::mutable_column_device_view` are also part of the kernel function
// signature. Packing the `mutable_column_device_view` parameters into a single struct resolves the
// issue.
struct fused_probes_output {
  cudf::mutable_column_device_view join_orders_lineitem_table_indices;
  cudf::mutable_column_device_view nation_table_indices;
  cudf::mutable_column_device_view customer_table_indices;
};

/**
 * @brief Fused probes kernel
 *
 * Probe `orders` joined with `lineitem` on c_custkey = o_custkey, then probe
 * `nation` on c_nationkey = n_nationkey. Output the row indices of the
 * matches.
 *
 * @param[in] o_custkey_multimap The hash multimap built on o_custkey.
 * @param[in] bloom_filter The Bloom filter built on o_custkey.
 * @param[in] n_nationkey_map The hash map built on n_nationkey.
 * @param[in] c_custkey_column Column for probing o_custkey_multimap.
 * @param[in] c_nationkey_column Column for probing n_nationkey_map.
 * @param[in, out] d_global_offset Global counter for number of output rows.
 * @param[out] out_indices Struct with build-side and probe-side row indices of matches.
 */
template <typename Identifier, typename MultimapRef, typename BloomFilterRef, typename MapRef>
__global__ void fused_probes_kernel(MultimapRef o_custkey_multimap,
                                    BloomFilterRef bloom_filter,
                                    MapRef n_nationkey_map,
                                    cudf::column_device_view c_custkey_column,
                                    cudf::column_device_view c_nationkey_column,
                                    cudf::size_type* d_global_offset,
                                    fused_probes_output out_indices);

}  // namespace q10
}  // namespace benchmark
}  // namespace gqe_python
