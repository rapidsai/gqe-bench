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

#include <gqe/physical/relation.hpp>

namespace gqe_python {
namespace benchmark {
namespace q21 {

/**
 * @brief Construct the left anti join probe relation.
 *
 * @param[in] left_table The left table.
 * @param[in] right_table The right table.
 */
std::shared_ptr<gqe::physical::relation> left_anti_join_probe(
  std::shared_ptr<gqe::physical::relation> left_table,
  std::shared_ptr<gqe::physical::relation> right_table);

/**
 * @brief Construct the left semi join probe relation.
 *
 * @param[in] left_table The left table.
 * @param[in] right_table The right table.
 */
std::shared_ptr<gqe::physical::relation> left_semi_join_probe(
  std::shared_ptr<gqe::physical::relation> left_table,
  std::shared_ptr<gqe::physical::relation> right_table);

/**
 * @brief Construct the scan tasks for left anti join retrieve from hash map.
 *
 * @param left_table The left table.
 * @param probe The probing relation that contains probe tasks and hash map.
 */
std::shared_ptr<gqe::physical::relation> left_anti_join_retrieve(
  std::shared_ptr<gqe::physical::relation> left_table,
  std::shared_ptr<gqe::physical::relation> probe);

/**
 * @brief Construct the scan tasks for left semi join retrieve from hash map.
 *
 * @param left_table The left table.
 * @param probe The probing relation that contains probe tasks and hash map.
 */
std::shared_ptr<gqe::physical::relation> left_semi_join_retrieve(
  std::shared_ptr<gqe::physical::relation> left_table,
  std::shared_ptr<gqe::physical::relation> probe);

}  // namespace q21
}  // namespace benchmark
}  // namespace gqe_python
