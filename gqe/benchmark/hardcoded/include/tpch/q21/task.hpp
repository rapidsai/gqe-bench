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
