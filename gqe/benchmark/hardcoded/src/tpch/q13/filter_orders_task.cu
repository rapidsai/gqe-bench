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

#include <tpch/q13/filter_orders_task.hpp>

#include <tpch/q13/kernels.cuh>
#include <utility/config.hpp>
#include <utility/utility.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/cuda.hpp>

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace gqe_python {
namespace benchmark {
namespace q13 {

namespace {

/**
 * @brief A task for the optimized orders table filter.
 */
class filter_orders_task : public gqe::task {
 public:
  /**
   * @brief Construct an orders table filter task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input_task The orders table read task.
   */
  filter_orders_task(gqe::context_reference ctx_ref,
                     int32_t task_id,
                     int32_t stage_id,
                     std::shared_ptr<gqe::task>&& input_task);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;
};

filter_orders_task::filter_orders_task(gqe::context_reference ctx_ref,
                                       int32_t task_id,
                                       int32_t stage_id,
                                       std::shared_ptr<gqe::task>&& input_task)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(input_task)}, {})
{
}

/**
 * @brief Functor to launch the filter kernel.
 */
struct filter_orders_functor {
  template <typename identifier_type>
  void operator()(gqe::context_reference ctx_ref,
                  cudf::column_device_view o_custkey,
                  cudf::column_device_view o_comment,
                  cudf::size_type* d_global_offset,
                  cudf::mutable_column_device_view out_o_custkey,
                  rmm::cuda_stream_view stream) const
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<identifier_type, int32_t> ||
                  std::is_same_v<identifier_type, int64_t>) {
      auto grid_size = gqe::utility::detect_launch_grid_size(
        ctx_ref._task_manager_context->get_device_properties(),
        filter_orders_kernel<identifier_type>,
        utility::block_dim,
        /* dynamic_shared_memory_bytes = */ 0);
      filter_orders_kernel<identifier_type><<<grid_size, utility::block_dim, 0, stream>>>(
        o_custkey, o_comment, d_global_offset, out_o_custkey);
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

void filter_orders_task::execute()
{
  auto const stream = rmm::cuda_stream_default;
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto orders_table    = dependent_tasks[0]->result().value();
  auto const row_count = orders_table.num_rows();

  auto o_custkey_column = cudf::column_device_view::create(orders_table.column(0), stream);
  auto o_comment_column = cudf::column_device_view::create(orders_table.column(1), stream);

  cudf::data_type identifier_type = o_custkey_column->type();

  auto out_o_custkey_column =
    cudf::make_numeric_column(identifier_type, row_count, cudf::mask_state::UNALLOCATED, stream);

  auto out_o_custkey_view =
    cudf::mutable_column_device_view::create(out_o_custkey_column->mutable_view(), stream);

  rmm::device_scalar<cudf::size_type> d_global_offset(0, stream);

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(identifier_type,
                        filter_orders_functor{},
                        get_context_reference(),
                        *o_custkey_column,
                        *o_comment_column,
                        d_global_offset.data(),
                        *out_o_custkey_view,
                        stream);

  GQE_CUDA_TRY(cudaGetLastError());

  cudf::size_type h_result_rows = d_global_offset.value(stream);
  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.reserve(1);

  auto o_custkey_result =
    std::make_unique<cudf::column>(cudf::slice(*out_o_custkey_column, {0, h_result_rows})[0]);
  result_columns.push_back(std::move(o_custkey_result));

  auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
  emit_result(std::move(result_table));

  remove_dependencies();
}

/**
 * @brief Generate the filter tasks.
 *
 * @param[in] children_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
std::vector<std::shared_ptr<gqe::task>> filter_orders_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id)
{
  std::vector<std::shared_ptr<gqe::task>> generated_tasks;
  generated_tasks.reserve(input_tasks[0].size());

  // Generate the filter tasks.
  for (auto& input_task : input_tasks[0]) {
    generated_tasks.push_back(
      std::make_shared<filter_orders_task>(ctx_ref, task_id, stage_id, std::move(input_task)));
    task_id++;
  }

  return generated_tasks;
}
}  // namespace

std::shared_ptr<gqe::physical::relation> filter_orders(
  std::shared_ptr<gqe::physical::relation> input)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {std::move(input)};

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, filter_orders_generate_tasks, false);
}

}  // namespace q13
}  // namespace benchmark
}  // namespace gqe_python
