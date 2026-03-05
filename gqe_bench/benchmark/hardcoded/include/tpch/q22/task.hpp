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

#include <gqe/executor/concatenate.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/utility/helpers.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/for_each.h>
#include <thrust/pair.h>

namespace gqe_python {
namespace benchmark {
namespace q22 {

/**
 * @brief A customized task to use a fused kernel for the project and filter operations.
 */
class fused_project_filter_task : public gqe::task {
 public:
  /**
   * @brief Construct a fused task of project and filter.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] customer_table Customer table as input to project and filter tasks.
   */
  fused_project_filter_task(gqe::context_reference ctx_ref,
                            int32_t task_id,
                            int32_t stage_id,
                            std::shared_ptr<gqe::task> customer_table);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;
};

/**
 * @brief Generate the output tasks from input tasks for fused kernel. It will concatenate the input
 * table.
 *
 * @param[in] children_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
std::vector<std::shared_ptr<gqe::task>> fused_project_filter_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id);

/**
 * @brief A customized task to use a fused kernel for the mark join between orders and customer.
 */
class mark_join_task : public gqe::task {
 public:
  /**
   * @brief Construct a customized task for mark join.
   * Mark join is used for left-anti join to reduce the bottleneck of a large build side.
   * Build side is customer table.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] customer_table Customer table as build side.
   * @param[in] orders_table Orders table as probe side.
   */
  mark_join_task(gqe::context_reference ctx_ref,
                 int32_t task_id,
                 int32_t stage_id,
                 std::shared_ptr<gqe::task> customer_table,
                 std::shared_ptr<gqe::task> orders_table);

  void execute() override;
};

/**
 * @brief Generate the output tasks from input tasks for mark join. It will concatenate the input
 * table.
 *
 * @param[in] children_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
std::vector<std::shared_ptr<gqe::task>> mark_join_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id);

/**
 * @brief Construct the fused project filter relation.
 *
 * @param[in] input Input relation.
 */
std::shared_ptr<gqe::physical::relation> fused_project_filter(
  std::shared_ptr<gqe::physical::relation> input);

/**
 * @brief Construct the mark join relation.
 *
 * @param[in] customer_table The customer table.
 * @param[in] orders_table The orders table.
 */
std::shared_ptr<gqe::physical::relation> mark_join(
  std::shared_ptr<gqe::physical::relation> customer_table,
  std::shared_ptr<gqe::physical::relation> orders_table);

}  // namespace q22
}  // namespace benchmark
}  // namespace gqe_python
