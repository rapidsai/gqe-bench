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

#include <tpch/q10/sort_limit_task.hpp>
#include <utility/config.hpp>
#include <utility/utility.hpp>

#include <cuda/std/utility>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/utility/cuda.hpp>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace gqe_python {
namespace benchmark {
namespace q10 {

namespace {

/* Subclasses of gqe::task */

/**
 * @brief A task for the sort-limit operation.
 */
class sort_limit_task : public gqe::task {
 public:
  /**
   * @brief Construct a sort-limit task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input_table The input table.
   * @param[in] limit Number of output rows.
   * @param[in] projection_indices Indices of columns to be included in the output table.
   */
  sort_limit_task(gqe::context_reference ctx_ref,
                  int32_t task_id,
                  int32_t stage_id,
                  std::shared_ptr<gqe::task> input_table,
                  const cudf::size_type key_col_idx,
                  const cudf::size_type limit,
                  std::vector<cudf::size_type>&& projection_indices);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  cudf::size_type _key_col_idx;
  cudf::size_type _limit;
  std::vector<cudf::size_type> _projection_indices;
};

/* Ctors for subclasses of gqe::task */

sort_limit_task::sort_limit_task(gqe::context_reference ctx_ref,
                                 int32_t task_id,
                                 int32_t stage_id,
                                 std::shared_ptr<gqe::task> input_table,
                                 const cudf::size_type key_col_idx,
                                 const cudf::size_type limit,
                                 std::vector<cudf::size_type>&& projection_indices)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(input_table)}, {}),
    _key_col_idx(key_col_idx),
    _limit(limit),
    _projection_indices(std::move(projection_indices))
{
}

/* Overrides for `execute()` method in subclasses of gqe::task. */

void sort_limit_task::execute()
{
  auto const stream = cudf::get_default_stream();
  prepare_dependencies();
  gqe::utility::nvtx_scoped_range task_range("sort_limit_task");
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto input_table = dependent_tasks[0]->result().value();

  auto top_k_indices =
    cudf::top_k_order(input_table.column(_key_col_idx), _limit, cudf::order::DESCENDING, stream);

  // Lambda for materializing the join output columns
  auto materialize_columns = [](cudf::table_view input_table,
                                std::vector<cudf::size_type>& column_indices,
                                cudf::column_view gather_map) {
    if (column_indices.empty()) { return std::vector<std::unique_ptr<cudf::column>>{}; }
    auto gathered_columns = cudf::gather(input_table.select(column_indices), gather_map)->release();
    return gathered_columns;
  };

  auto output_columns =
    materialize_columns(input_table, _projection_indices, top_k_indices->view());

  auto result_table = std::make_unique<cudf::table>(std::move(output_columns));
  emit_result(std::move(result_table));

  remove_dependencies();
}

/* Functors for generating tasks. Used by interface functions. */

/**
 * @brief Functor for generating sort-limit tasks.
 *
 * @param[in] input_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
struct sort_limit_generate_tasks {
  cudf::size_type key_col_idx;
  cudf::size_type limit;
  std::vector<cudf::size_type> projection_indices;

  std::vector<std::shared_ptr<gqe::task>> operator()(
    std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
    gqe::context_reference ctx_ref,
    int32_t& task_id,
    int32_t stage_id)
  {
    assert(input_tasks.size() == 1 && "expected exactly one input relation");

    std::shared_ptr<gqe::task> input_table;
    // Don't bother concatenating if the number of row groups is only one.
    if (input_tasks[0].size() == 1) {
      input_table = input_tasks[0][0];
    } else {
      input_table =
        std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, input_tasks[0]);
      task_id++;
    }

    return {std::make_shared<sort_limit_task>(
      ctx_ref, task_id, stage_id, input_table, key_col_idx, limit, std::move(projection_indices))};
  }
};

}  // namespace

/* Interface function definitions */

std::shared_ptr<gqe::physical::relation> sort_limit(
  std::shared_ptr<gqe::physical::relation> input_table,
  const cudf::size_type key_col_idx,
  const cudf::size_type limit,
  std::vector<cudf::size_type> projection_indices)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {std::move(input_table)};

  sort_limit_generate_tasks sort_limit_task_generator{
    key_col_idx, limit, std::move(projection_indices)};

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, sort_limit_task_generator, /* last_child_break_pipeline = */ false);
}

}  // namespace q10
}  // namespace benchmark
}  // namespace gqe_python
