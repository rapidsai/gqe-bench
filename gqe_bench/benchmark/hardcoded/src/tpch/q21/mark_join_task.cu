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

#include <tpch/q21/mark_join.cuh>
#include <tpch/q21/task.hpp>
#include <utility/hash_map_cache.hpp>
#include <utility/utility.hpp>

#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/copying.hpp>

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
namespace q21 {

auto constexpr enable_left_semi_bf = true;
auto constexpr enable_left_anti_bf = true;

/**
 * @brief A customized task to perform left-anti join with mark join, and it also fuses a filter
 * kernel inside.
 */
template <bool is_anti_join>
class mark_join_probe_task : public gqe::task {
 public:
  /**
   * @brief Construct a left anti join task that fuses a filter operator.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] build_table The build table, used for join predicate conditions.
   * @param[in] probe_table The probe table, used for probing against the hash table.
   * @param[in] hash_map_cache Shared hash map among different row groups.
   */
  mark_join_probe_task(gqe::context_reference ctx_ref,
                       int32_t task_id,
                       int32_t stage_id,
                       std::shared_ptr<gqe::task> build_table,
                       std::shared_ptr<gqe::task> probe_table,
                       std::shared_ptr<utility::task_hash_map> hash_map_cache);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

  /**
   * @brief Return the hash map.
   */
  std::shared_ptr<utility::task_hash_map> hash_map();

 private:
  std::shared_ptr<utility::task_hash_map> _hash_map_cache;
};

/**
 * @brief A customized task to scan the hash table and emit the marked/unmarked results.
 */
template <bool is_anti_join>
class mark_join_scan_task : public gqe::task {
 public:
  /**
   * @brief Construct a left anti join scan task that outputs marked/unmarked entries.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] children_task Children tasks.
   * @param[in] hash_map_cache Shared hash map among different row groups.
   */
  mark_join_scan_task(gqe::context_reference ctx_ref,
                      int32_t task_id,
                      int32_t stage_id,
                      std::vector<std::shared_ptr<gqe::task>> children_task,
                      std::shared_ptr<utility::task_hash_map> _hash_map_cache);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

  /**
   * @brief Return the supplier hash map.
   */
  std::shared_ptr<utility::task_hash_map> hash_map();

 private:
  std::shared_ptr<utility::task_hash_map> _hash_map_cache;
};

template <bool is_anti_join>
mark_join_probe_task<is_anti_join>::mark_join_probe_task(
  gqe::context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<gqe::task> build_table,
  std::shared_ptr<gqe::task> probe_table,
  std::shared_ptr<utility::task_hash_map> hash_map_cache)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(build_table), std::move(probe_table)}, {}),
    _hash_map_cache(std::move(hash_map_cache))
{
}

template <bool is_anti_join>
std::shared_ptr<utility::task_hash_map> mark_join_probe_task<is_anti_join>::hash_map()
{
  return _hash_map_cache;
}

/**
 * @brief Insertion functor to insert the build keys into the hash map and bloom filter.
 */
template <typename Identifier, bool enable_bf = true>
struct mark_join_insertion_functor {
  rmm::cuda_stream_view main_stream;
  cudf::column_device_view build_keys;

  void operator()(mark_join_map_type<Identifier>& hash_map,
                  utility::bloom_filter_type<Identifier>& bloom_filter) const
  {
    auto map_ref = hash_map.ref(cuco::op::find, cuco::op::for_each, cuco::op::insert);
    auto const build_table_size = build_keys.size();
    thrust::for_each(thrust::make_counting_iterator<cudf::size_type>(0),
                     thrust::make_counting_iterator<cudf::size_type>(build_table_size),
                     [map_ref, keys = build_keys] __device__(auto row_idx) mutable {
                       Identifier key = static_cast<Identifier>(keys.element<Identifier>(row_idx));
                       map_ref.insert(thrust::pair<Identifier, cudf::size_type>(key, row_idx));
                     });
    if constexpr (enable_bf) {
      auto it = thrust::make_transform_iterator(
        thrust::make_counting_iterator<cudf::size_type>(0),
        [keys = build_keys] __device__(auto row_idx) -> Identifier {
          return static_cast<Identifier>(keys.element<Identifier>(row_idx));
        });
      bloom_filter.add(it, it + build_table_size, main_stream);
    }
  }
};

/**
 * @brief Functor to build the hash table for mark join.
 */
template <bool enable_bf = true>
struct hash_table_build_functor {
  rmm::cuda_stream_view main_stream;
  cudf::column_device_view build_keys;
  utility::task_hash_map& hash_map_cache;

  template <typename Identifier>
  void operator()() const
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      auto const build_table_size                 = build_keys.size();
      auto constexpr load_factor                  = 0.5;
      auto constexpr bf_size_factor               = 2;
      cuco::extent<std::size_t> num_filter_blocks = 0;
      if constexpr (enable_bf) {
        num_filter_blocks =
          utility::get_bloom_filter_blocks<gqe_python::utility::bloom_filter_type<Identifier>>(
            build_table_size * bf_size_factor);
      }
      hash_map_cache
        .create_map_bf_and_insert<Identifier, cudf::size_type, mark_join_map_type<Identifier>>(
          build_table_size,
          load_factor,
          num_filter_blocks,
          mark_join_insertion_functor<Identifier, enable_bf>(main_stream, build_keys));
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

/**
 * @brief Functor to launch the mark join kernel.
 *
 */
struct left_anti_join_probe_functor {
  rmm::cuda_stream_view main_stream;
  cudf::table_view left_table;
  cudf::table_view right_table;
  utility::task_hash_map& hash_map_cache;

  template <typename Identifier>
  void operator()() const
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      // Determine the hash table size.
      auto const left_table_size = left_table.num_rows();

      auto left_suppkey_column      = cudf::column_device_view::create(left_table.column(0));
      auto left_orderkey_column     = cudf::column_device_view::create(left_table.column(1));
      auto right_suppkey_column     = cudf::column_device_view::create(right_table.column(0));
      auto right_orderkey_column    = cudf::column_device_view::create(right_table.column(1));
      auto right_receiptdate_column = cudf::column_device_view::create(right_table.column(2));
      auto right_commitdate_column  = cudf::column_device_view::create(right_table.column(3));

      auto& map =
        hash_map_cache.get_map<Identifier, cudf::size_type, mark_join_map_type<Identifier>>();
      auto map_ref = map.ref(cuco::find, cuco::for_each, cuco::op::insert);

      auto& bloom_filter    = hash_map_cache.get_bloom_filter<Identifier>();
      auto bloom_filter_ref = bloom_filter.ref();

      // Global counters.
      rmm::device_scalar<cudf::size_type> d_probe_counter(0, main_stream);
      cudf::size_type* d_probe_counter_ptr = d_probe_counter.data();

      left_anti_join_params<Identifier> params{map_ref,
                                               bloom_filter_ref,
                                               *left_suppkey_column,
                                               *right_receiptdate_column,
                                               *right_commitdate_column,
                                               *right_suppkey_column,
                                               *right_orderkey_column,
                                               d_probe_counter_ptr};

      auto probe_grid_size =
        gqe::utility::detect_launch_grid_size(left_anti_join_kernel<Identifier>,
                                              utility::block_dim,
                                              /* dynamic_shared_memory_bytes = */ 0);
      left_anti_join_kernel<Identifier>
        <<<probe_grid_size, utility::block_dim, 0, main_stream>>>(params);
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

/**
 * @brief Functor to launch the mark join kernel.
 *
 */
struct left_semi_join_probe_functor {
  rmm::cuda_stream_view main_stream;
  cudf::table_view left_table;
  cudf::table_view right_table;
  utility::task_hash_map& hash_map_cache;

  template <typename Identifier>
  void operator()() const
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      // Determine the hash table size.
      auto const left_table_size = left_table.num_rows();

      auto left_suppkey_column   = cudf::column_device_view::create(left_table.column(0));
      auto left_orderkey_column  = cudf::column_device_view::create(left_table.column(1));
      auto right_suppkey_column  = cudf::column_device_view::create(right_table.column(0));
      auto right_orderkey_column = cudf::column_device_view::create(right_table.column(1));

      auto& map =
        hash_map_cache.get_map<Identifier, cudf::size_type, mark_join_map_type<Identifier>>();
      auto map_ref = map.ref(cuco::find, cuco::for_each, cuco::op::insert);

      auto& bloom_filter    = hash_map_cache.get_bloom_filter<Identifier>();
      auto bloom_filter_ref = bloom_filter.ref();

      // Global counters.
      rmm::device_scalar<cudf::size_type> d_probe_counter(0, main_stream);
      cudf::size_type* d_probe_counter_ptr = d_probe_counter.data();

      left_semi_join_params<Identifier> params{map_ref,
                                               bloom_filter_ref,
                                               *left_suppkey_column,
                                               *right_suppkey_column,
                                               *right_orderkey_column,
                                               d_probe_counter_ptr};

      auto probe_grid_size = gqe_python::utility::find_grid_size(left_semi_join_kernel<Identifier>,
                                                                 gqe_python::utility::block_dim);
      left_semi_join_kernel<Identifier>
        <<<probe_grid_size, gqe_python::utility::block_dim, 0, main_stream>>>(params);
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

template <bool is_anti_join>
void mark_join_probe_task<is_anti_join>::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 2 && "mark_join_probe expected exactly two input relations");

  auto const main_stream = cudf::get_default_stream();
  gqe::utility::nvtx_scoped_range mark_join_probe_task_range("q21_mark_join_probe_task");

  // Get the input tables.
  auto left_table                 = dependent_tasks[0]->result().value();
  auto right_table                = dependent_tasks[1]->result().value();
  auto const left_table_size      = left_table.num_rows();
  auto build_orderkey_column      = cudf::column_device_view::create(left_table.column(1));
  cudf::data_type identifier_type = left_table.column(1).type();

  // Build the hash table.
  auto constexpr enable_bf = is_anti_join ? enable_left_anti_bf : enable_left_semi_bf;
  cudf::type_dispatcher(
    identifier_type,
    hash_table_build_functor<enable_bf>{main_stream, *build_orderkey_column, *_hash_map_cache});
  GQE_CUDA_TRY(cudaGetLastError());

  // The result size is capped by the size of left table.
  rmm::device_uvector<cudf::size_type> left_out_indices(left_table_size, main_stream);

  rmm::device_scalar<cudf::size_type> d_scan_counter(0, main_stream);
  cudf::size_type* d_scan_counter_ptr = d_scan_counter.data();

  // Probe the hash table.
  if constexpr (is_anti_join) {
    cudf::type_dispatcher(
      identifier_type,
      left_anti_join_probe_functor{main_stream, left_table, right_table, *_hash_map_cache});
  } else {
    cudf::type_dispatcher(
      identifier_type,
      left_semi_join_probe_functor{main_stream, left_table, right_table, *_hash_map_cache});
  }

  GQE_CUDA_TRY(cudaGetLastError());
  main_stream.synchronize();

  // Emit an empty result table.
  std::vector<std::unique_ptr<cudf::column>> out_columns;
  emit_result(std::make_unique<cudf::table>(std::move(out_columns)));
  remove_dependencies();
}

template <bool is_anti_join>
mark_join_scan_task<is_anti_join>::mark_join_scan_task(
  gqe::context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::vector<std::shared_ptr<gqe::task>> children_task,
  std::shared_ptr<utility::task_hash_map> hash_map_cache)
  : gqe::task(ctx_ref, task_id, stage_id, std::move(children_task), {}),
    _hash_map_cache(std::move(hash_map_cache))
{
}

template <bool is_anti_join>
std::shared_ptr<utility::task_hash_map> mark_join_scan_task<is_anti_join>::hash_map()
{
  return _hash_map_cache;
}

/**
 * @brief Functor to scan the hash map and output the left table indices that are marked/unmarked.
 */
template <bool is_anti_join>
struct mark_join_scan_functor {
  rmm::cuda_stream_view main_stream;
  utility::task_hash_map& hash_map_cache;
  cudf::size_type* d_scan_counter_ptr;
  cudf::size_type* left_out_indices;

  template <typename Identifier>
  void operator()() const
  {
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      auto& map =
        hash_map_cache.get_map<Identifier, cudf::size_type, mark_join_map_type<Identifier>>();
      auto map_device_view = map.ref(cuco::op::find, cuco::op::for_each, cuco::op::insert);
      auto scan_grid_size =
        gqe::utility::detect_launch_grid_size(iterate_join_map<is_anti_join, Identifier>,
                                              utility::block_dim,
                                              /* dynamic_shared_memory_bytes = */ 0);
      iterate_join_map<is_anti_join, Identifier>
        <<<scan_grid_size, gqe_python::utility::block_dim, 0, main_stream>>>(
          map_device_view, left_out_indices, d_scan_counter_ptr);
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

template <bool is_anti_join>
void mark_join_scan_task<is_anti_join>::execute()
{
  auto const main_stream = cudf::get_default_stream();
  prepare_dependencies();
  gqe::utility::nvtx_scoped_range mark_join_scan_task_range("q21_mark_join_scan_task");
  auto dependent_tasks = dependencies();
  // Get the input tables
  auto left_table            = dependencies()[0]->result().value();
  auto const left_table_size = left_table.num_rows();

  cudf::data_type identifier_type = left_table.column(0).type();
  // The result size is capped by the size of left table.
  rmm::device_uvector<cudf::size_type> left_out_indices(left_table_size, main_stream);

  rmm::device_scalar<cudf::size_type> d_scan_counter(0, main_stream);
  cudf::size_type* d_scan_counter_ptr = d_scan_counter.data();

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(
    identifier_type,
    mark_join_scan_functor<is_anti_join>{
      main_stream, *_hash_map_cache, d_scan_counter_ptr, left_out_indices.data()});

  GQE_CUDA_TRY(cudaGetLastError());
  main_stream.synchronize();

  std::size_t h_counter = d_scan_counter.value(main_stream);

  // Materialize the columns.
  // Materialize the join output
  auto materialize_column =
    [](cudf::table_view input_table, cudf::size_type column_idx, cudf::column_view gather_map) {
      auto gathered_column = cudf::gather(input_table.select({column_idx}), gather_map)->release();
      return std::move(gathered_column[0]);
    };
  cudf::column_view gather_map{cudf::data_type{cudf::type_id::INT32},
                               static_cast<cudf::size_type>(h_counter),
                               static_cast<void const*>(left_out_indices.data()),
                               nullptr,
                               0};
  std::vector<std::unique_ptr<cudf::column>> out_columns;
  if constexpr (is_anti_join) {
    // Materialize l_suppkey, l_orderkey, s_name.
    out_columns.push_back(materialize_column(left_table, 0, gather_map));
    out_columns.push_back(materialize_column(left_table, 1, gather_map));
    out_columns.push_back(materialize_column(left_table, 2, gather_map));
  } else {
    // Only materialize s_name.
    out_columns.push_back(materialize_column(left_table, 2, gather_map));
  }
  emit_result(std::make_unique<cudf::table>(std::move(out_columns)));
  remove_dependencies();
}

/**
 * @brief Functor for generating the output tasks from input tasks for left {anti/semi} join kernel.
 *
 * @param[in] children_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
template <bool is_anti_join>
std::vector<std::shared_ptr<gqe::task>> mark_join_probe_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id)
{
  assert(children_tasks.size() == 2 && "mark_join_probe expected exactly two input relation");
  std::shared_ptr<gqe::task> left_table;
  // Don't bother concatenating if the number of row groups is only one.
  // But we need to concatenate the left table as build side.
  if (children_tasks[0].size() == 1) {
    left_table = children_tasks[0][0];
  } else {
    left_table =
      std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[0]);
    task_id++;
  }

  auto constexpr enable_bf = is_anti_join ? enable_left_anti_bf : enable_left_semi_bf;
  // Create the hash map cache for the join, the parameters would be overwritten during task
  // execution.
  auto hash_map_cache = std::make_shared<utility::task_hash_map>(
    /*cardinality_estimate=*/-1, /*load_factor=*/0.5, /*use_bloom_filter=*/enable_bf);

  // We need to deal with different partition for the right-hand side large table.
  std::vector<std::shared_ptr<gqe::task>> pipeline_results;

  for (const auto& right_task : children_tasks[1]) {
    pipeline_results.push_back(std::make_shared<mark_join_probe_task<is_anti_join>>(
      ctx_ref, task_id, stage_id, left_table, right_task, hash_map_cache));
    task_id++;
  }

  return pipeline_results;
}

/**
 * @brief Functor for generating the output tasks from probing tasks from left {anti/semi} join
 * kernel.
 *
 * @param[in] children_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
template <bool is_anti_join>
std::vector<std::shared_ptr<gqe::task>> mark_join_retrieve_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id)
{
  assert(children_tasks.size() == 2 && "mark_join_retrieve expected exactly two input relation");

  std::shared_ptr<gqe::task> left_table;
  // Don't bother concatenating if the number of row groups is only one.
  // But we need to concatenate the left table as build side.
  // Note: we would concatenate the left table again, the previous one is in build and probe step.
  if (children_tasks[0].size() == 1) {
    left_table = children_tasks[0][0];
  } else {
    left_table =
      std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[0]);
    task_id++;
  }

  auto probe_task = dynamic_cast<mark_join_probe_task<is_anti_join>*>(children_tasks[1][0].get());
  assert(probe_task != nullptr && "Input task expects mark_join_probe_task");
  auto hash_map_cache = probe_task->hash_map();
  std::shared_ptr<gqe::task> concatenated_probe;
  // Don't bother concatenating if the number of row groups is only one.
  if (children_tasks[1].size() == 1) {
    concatenated_probe = children_tasks[1][0];
  } else {
    concatenated_probe =
      std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[1]);
    task_id++;
  }
  std::shared_ptr<gqe::task> scan_task = std::make_shared<mark_join_scan_task<is_anti_join>>(
    ctx_ref,
    task_id,
    stage_id,
    std::vector<std::shared_ptr<gqe::task>>{left_table, concatenated_probe},
    hash_map_cache);
  task_id++;
  std::vector<std::shared_ptr<gqe::task>> results;
  results.push_back(scan_task);
  return results;
}

std::shared_ptr<gqe::physical::relation> left_anti_join_probe(
  std::shared_ptr<gqe::physical::relation> left_table,
  std::shared_ptr<gqe::physical::relation> right_table)
{
  return std::make_shared<gqe::physical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(
      {std::move(left_table), std::move(right_table)}),
    mark_join_probe_generate_tasks</*is_anti_join=*/true>,
    /*last_child_break_pipeline=*/false);
}

std::shared_ptr<gqe::physical::relation> left_anti_join_retrieve(
  std::shared_ptr<gqe::physical::relation> left_table,
  std::shared_ptr<gqe::physical::relation> probe)
{
  return std::make_shared<gqe::physical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(
      {std::move(left_table), std::move(probe)}),
    mark_join_retrieve_generate_tasks</*is_anti_join=*/true>,
    /*last_child_break_pipeline=*/true);
}

std::shared_ptr<gqe::physical::relation> left_semi_join_probe(
  std::shared_ptr<gqe::physical::relation> left_table,
  std::shared_ptr<gqe::physical::relation> right_table)
{
  return std::make_shared<gqe::physical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(
      {std::move(left_table), std::move(right_table)}),
    mark_join_probe_generate_tasks</*is_anti_join=*/false>,
    /*last_child_break_pipeline=*/false);
}

std::shared_ptr<gqe::physical::relation> left_semi_join_retrieve(
  std::shared_ptr<gqe::physical::relation> left_table,
  std::shared_ptr<gqe::physical::relation> probe)
{
  return std::make_shared<gqe::physical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(
      {std::move(left_table), std::move(probe)}),
    mark_join_retrieve_generate_tasks</*is_anti_join=*/false>,
    /*last_child_break_pipeline=*/true);
}

}  // namespace q21
}  // namespace benchmark
}  // namespace gqe_python
