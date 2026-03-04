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

#include <cudf/types.hpp>

#include <gqe/physical/relation.hpp>

#include <memory>

namespace gqe_python {
namespace benchmark {
namespace q10 {

/**
 * @brief Build multimap for inner join.
 *
 * @param[in] build_side_table The build-side table.
 * @param[in] key_column_idx Index of key column in build-side table.
 */
std::shared_ptr<gqe::physical::relation> fused_probes_join_multimap_build(
  std::shared_ptr<gqe::physical::relation> build_side_table, const cudf::size_type key_column_idx);

/**
 * @brief Build map for inner join.
 *
 * @param[in] build_side_table The build-side table.
 * @param[in] key_column_idx Index of key column in build-side table.
 */
std::shared_ptr<gqe::physical::relation> fused_probes_join_map_build(
  std::shared_ptr<gqe::physical::relation> build_side_table, const cudf::size_type key_column_idx);

/**
 * @brief Fused probes of two joins for Q10.
 *
 * After the first join in Q10 of `orders` and `lineitem` on
 * `o_orderkey = l_orderkey`, a multimap is built for the output on
 * `o_custkey`, and a map for the `nation` table on `n_nationkey`. These are
 * then used as inputs to this function which first probes
 * `o_custkey_to_row_indices_multimap` with `c_custkey`. For each of the
 * matches in the first probe, `c_nationkey` is then used to probe
 * `n_nationkey_to_row_index_map`.
 *
 * @param[in] o_custkey_to_row_indices_multimap Multimap built on `o_custkey`
 * in the output of the join between `orders` and `lineitem`.
 * @param[in] n_nationkey_to_row_index_map Map built on `n_nationkey` in `nation`.
 * @param[in] join_orders_lineitem_table Join output between `orders` and `lineitem`.
 * @param[in] nation_table The `nation` table.
 * @param[in] customer_table The `customer` table.
 */
std::shared_ptr<gqe::physical::relation> fused_probes_join_probe(
  std::shared_ptr<gqe::physical::relation> o_custkey_to_row_indices_multimap,
  std::shared_ptr<gqe::physical::relation> n_nationkey_to_row_index_map,
  std::shared_ptr<gqe::physical::relation> join_orders_lineitem_table,
  std::shared_ptr<gqe::physical::relation> nation_table,
  std::shared_ptr<gqe::physical::relation> customer_table);

}  // namespace q10
}  // namespace benchmark
}  // namespace gqe_python
