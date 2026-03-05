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

#include <tpch/q16/fused_filter_join.cuh>
#include <tpch/q16/task.hpp>
#include <utility/utility.hpp>

#include <gqe/executor/aggregate.hpp>
#include <gqe/executor/groupby.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/groupby.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace gqe_python {
namespace benchmark {
namespace q16 {

// This may not be used, just for reference that the encoding and decoding are possible for strings.
[[maybe_unused]] constexpr int32_t num_distinct_p_brand = 25;
[[maybe_unused]] constexpr int32_t max_len_p_brand      = 8;
[[maybe_unused]] constexpr int32_t num_distinct_p_type  = 150;
[[maybe_unused]] constexpr int32_t max_len_p_type       = 25;

/**
 * @brief A customized task to have an efficient implementation of groupby.
 */
class aggregate_task : public gqe::task {
 public:
  /**
   * @brief Construct a customized aggregate task for COUNT(DISTINCT).
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input_table Joined supplier, part and partsupp table as input to aggregation.
   */
  aggregate_task(gqe::context_reference ctx_ref,
                 int32_t task_id,
                 int32_t stage_id,
                 std::shared_ptr<gqe::task> input_table);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;
};

aggregate_task::aggregate_task(gqe::context_reference ctx_ref,
                               int32_t task_id,
                               int32_t stage_id,
                               std::shared_ptr<gqe::task> input_table)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(input_table)}, {})
{
}

/**
 * @brief This is a deprecated implementation that uses dictionary encoding and decoding to reduce
 * string operation overheads in sorting and groupby.
 */

// void aggregate_task::execute()
// {
//   auto const main_stream = cudf::get_default_stream();
//   prepare_dependencies();
//   gqe::utility::nvtx_scoped_range aggregate_task_range("q16_aggregate_task");
//   auto dependent_tasks = dependencies();
//   assert(dependent_tasks.size() == 1);
//   auto input_table     = dependent_tasks[0]->result().value();
//   auto const row_count = static_cast<cudf::size_type>(input_table.num_rows());

//   // 1. Dictionary encode brand and type
//   auto p_brand_col    = input_table.column(0);
//   auto p_type_col     = input_table.column(1);
//   auto p_size_col     = input_table.column(2);
//   auto ps_suppkey_col = input_table.column(3);

//   auto dict_brand = cudf::dictionary::encode(p_brand_col);
//   auto dict_type  = cudf::dictionary::encode(p_type_col);

//   auto encode_brand_col = cudf::dictionary_column_view(dict_brand->view());
//   auto brand_codes      = encode_brand_col.indices();
//   auto brand_keys       = encode_brand_col.keys();

//   auto encode_type_col = cudf::dictionary_column_view(dict_type->view());
//   auto type_codes      = encode_type_col.indices();
//   auto type_keys       = encode_type_col.keys();

//   cudf::column_view active_mask;
//   gqe::groupby::groupby gb1(
//     cudf::table_view({brand_codes, type_codes, p_size_col, ps_suppkey_col}));
//   std::vector<cudf::groupby::aggregation_request> requests1;
//   auto [first_key_outputs, first_agg_results] =
//     gb1.aggregate(requests1, active_mask);

//   gqe::groupby::groupby gb2(cudf::table_view({first_key_outputs->view().column(0),
//                                               first_key_outputs->view().column(1),
//                                               first_key_outputs->view().column(2)}));
//   std::vector<cudf::groupby::aggregation_request> requests2(1);
//   requests2[0].values = first_key_outputs->view().column(0);
//   requests2[0].aggregations.push_back(
//     cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
//   auto [second_key_outputs, second_agg_results] =
//     gb2.aggregate(requests2, active_mask);

//   std::vector<std::unique_ptr<cudf::column>> aggregate_keys = second_key_outputs->release();
//   auto grouped_brand_codes                                  = aggregate_keys[0]->view();
//   auto grouped_type_codes                                   = aggregate_keys[1]->view();
//   auto grouped_brands = cudf::make_dictionary_column(brand_keys, grouped_brand_codes,
//   main_stream); auto grouped_types  = cudf::make_dictionary_column(type_keys, grouped_type_codes,
//   main_stream);

//   cudf::dictionary_column_view decode_dict_brand(grouped_brands->view());
//   cudf::dictionary_column_view decode_dict_type(grouped_types->view());

//   std::unique_ptr<cudf::column> brand_decoded = cudf::dictionary::decode(decode_dict_brand);
//   std::unique_ptr<cudf::column> type_decoded  = cudf::dictionary::decode(decode_dict_type);

//   std::vector<std::unique_ptr<cudf::column>> out_columns;
//   out_columns.push_back(std::move(brand_decoded));                     // p_brand
//   out_columns.push_back(std::move(type_decoded));                      // p_type
//   out_columns.push_back(std::move(aggregate_keys[2]));                 // p_size
//   out_columns.push_back(std::move(second_agg_results[0].results[0]));  // supplier_cnt
//   emit_result(std::make_unique<cudf::table>(std::move(out_columns)));
//   remove_dependencies();
// }

void aggregate_task::execute()
{
  auto const main_stream = cudf::get_default_stream();
  prepare_dependencies();
  gqe::utility::nvtx_scoped_range aggregate_task_range("q16_aggregate_task");
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto input_table = dependent_tasks[0]->result().value();

  auto p_brand_col    = input_table.column(0);
  auto p_type_col     = input_table.column(1);
  auto p_size_col     = input_table.column(2);
  auto ps_suppkey_col = input_table.column(3);

  cudf::column_view active_mask;
  // First groupby on (p_brand, p_type, p_size, ps_suppkey)
  gqe::groupby::groupby gb1(
    cudf::table_view({p_brand_col, p_type_col, p_size_col, ps_suppkey_col}));
  std::vector<cudf::groupby::aggregation_request> requests1;
  auto [first_key_outputs, first_agg_results] = gb1.aggregate(requests1, active_mask);

  // Second groupby on (p_brand, p_type, p_size) with COUNT(ps_suppkey), this would yield DISTINCT
  // for ps_suppkey.
  gqe::groupby::groupby gb2(cudf::table_view({first_key_outputs->view().column(0),
                                              first_key_outputs->view().column(1),
                                              first_key_outputs->view().column(2)}));
  std::vector<cudf::groupby::aggregation_request> requests2(1);
  requests2[0].values = first_key_outputs->view().column(0);
  requests2[0].aggregations.push_back(
    cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
  auto [second_key_outputs, second_agg_results] = gb2.aggregate(requests2, active_mask);

  std::vector<std::unique_ptr<cudf::column>> out_columns = second_key_outputs->release();
  out_columns.push_back(std::move(second_agg_results[0].results[0]));

  emit_result(std::make_unique<cudf::table>(std::move(out_columns)));
  remove_dependencies();
}

/**
 * @brief Generate the tasks for aggregate relation. We concatenate the input tables if there are
 * more than 1 partition.
 *
 * @param[in] children_tasks children tasks
 * @param[in] ctx_ref context reference
 * @param[in] task_id task ID
 * @param[in] stage_id stage ID
 */
std::vector<std::shared_ptr<gqe::task>> aggregate_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id)
{
  std::shared_ptr<gqe::task> input_table;
  if (children_tasks[0].size() == 1) {
    input_table = children_tasks[0][0];
  } else {
    input_table =
      std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[0]);
    task_id++;
  }
  std::vector<std::shared_ptr<gqe::task>> pipeline_results;
  pipeline_results.push_back(
    std::make_shared<aggregate_task>(ctx_ref, task_id, stage_id, input_table));
  task_id++;
  return pipeline_results;
}

std::shared_ptr<gqe::physical::relation> aggregate(
  std::shared_ptr<gqe::physical::relation> input_table)
{
  // Here need to break the pipeline, otherwise the aggregate task would be put into the same stage
  // as join tasks, and lead to only one root task that doesn't allow multiple workers.
  return std::make_shared<gqe::physical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>({std::move(input_table)}),
    aggregate_generate_tasks,
    /*last_child_break_pipeline=*/true);
}

}  // namespace q16
}  // namespace benchmark
}  // namespace gqe_python
