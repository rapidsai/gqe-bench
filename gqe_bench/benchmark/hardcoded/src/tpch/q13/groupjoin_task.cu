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

#include <tpch/q13/groupjoin_task.hpp>

#include <tpch/q13/kernels.cuh>
#include <utility/config.hpp>
#include <utility/hash_map_cache.hpp>
#include <utility/utility.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
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

template <typename T>
using groupjoin_map_type = utility::map_type<T, cudf::size_type>;

/**
 * @brief A task for the groupjoin build.
 */
class groupjoin_build_task : public gqe::task {
 public:
  /**
   * @brief Construct a groupjoin build task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input_task The orders table read task.
   * @param[in] hash_map The hash map shared among tasks.
   */
  groupjoin_build_task(gqe::context_reference ctx_ref,
                       int32_t task_id,
                       int32_t stage_id,
                       std::shared_ptr<gqe::task>&& input_task,
                       std::shared_ptr<utility::task_hash_map> hash_map);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

  /**
   * @brief Return the hash map.
   */
  std::shared_ptr<utility::task_hash_map> hash_map();

 private:
  std::shared_ptr<utility::task_hash_map> _hash_map;
};

/**
 * @brief A task for the groupjoin probe.
 */
class groupjoin_probe_task : public gqe::task {
 public:
  /**
   * @brief Construct a groupjoin probe task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] groupjoin_build_task The concatenated build tasks of the groupjoin.
   * @param[in] orders_task The probe-side row group task, from the orders table.
   * @param[in] hash_map The hash map shared among tasks.
   * @param[in] use_fused_kernel Perform the fused filter and probe instead of only the probe.
   */
  groupjoin_probe_task(gqe::context_reference ctx_ref,
                       int32_t task_id,
                       int32_t stage_id,
                       std::shared_ptr<gqe::task> groupjoin_build_task,
                       std::shared_ptr<gqe::task>&& orders_task,
                       std::shared_ptr<utility::task_hash_map> hash_map,
                       bool use_fused_kernel);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

  /**
   * @brief Return the hash map.
   */
  std::shared_ptr<utility::task_hash_map> hash_map();

 private:
  std::shared_ptr<utility::task_hash_map> _hash_map;
  bool _use_fused_kernel;
};

/**
 * @brief A task for the groupjoin retrieve.
 */
class groupjoin_retrieve_task : public gqe::task {
 public:
  /**
   * @brief Construct an groupjoin retrieve task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] groupjoin_probe_task The concatenated probe tasks.
   * @param[in] hash_map The hash map shared among tasks.
   */
  groupjoin_retrieve_task(gqe::context_reference ctx_ref,
                          int32_t task_id,
                          int32_t stage_id,
                          std::shared_ptr<gqe::task> groupjoin_probe_task,
                          std::shared_ptr<utility::task_hash_map> hash_map);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::shared_ptr<utility::task_hash_map> _hash_map;
};

groupjoin_build_task::groupjoin_build_task(gqe::context_reference ctx_ref,
                                           int32_t task_id,
                                           int32_t stage_id,
                                           std::shared_ptr<gqe::task>&& input_task,
                                           std::shared_ptr<utility::task_hash_map> hash_map)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(input_task)}, {}),
    _hash_map(std::move(hash_map))
{
}

/**
 * @brief Functor to launch the groupjoin build kernel.
 */
struct groupjoin_build_functor {
  template <typename Identifier>
  void operator()(cudf::column_device_view c_custkey,
                  utility::task_hash_map& hash_map_wrapper,
                  rmm::cuda_stream_view)
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      // Lazy thread-safe instantiation of CuCo hash map.
      auto& hash_map =
        hash_map_wrapper.get_map<Identifier, cudf::size_type, groupjoin_map_type<Identifier>>();

      thrust::for_each(
        thrust::make_counting_iterator<cudf::size_type>(0),
        thrust::make_counting_iterator<cudf::size_type>(c_custkey.size()),
        [map = hash_map.ref(cuco::insert), keys = c_custkey] __device__(auto row_idx) mutable {
          // We don't need to check for NULLs here.
          auto key = keys.element<Identifier>(row_idx);
          map.insert(cuco::pair<Identifier, cudf::size_type>(key, 0));
        });
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

void groupjoin_build_task::execute()
{
  auto const stream = rmm::cuda_stream_default;
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto customer_table   = dependent_tasks[0]->result().value();
  auto c_custkey_column = cudf::column_device_view::create(customer_table.column(0), stream);

  cudf::data_type identifier_type = c_custkey_column->type();

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(
    identifier_type, groupjoin_build_functor{}, *c_custkey_column, *_hash_map, stream);

  GQE_CUDA_TRY(cudaGetLastError());

  auto result_table = std::make_unique<cudf::table>();
  emit_result(std::move(result_table));

  remove_dependencies();
}

std::shared_ptr<utility::task_hash_map> groupjoin_build_task::hash_map() { return _hash_map; }

/**
 * @brief Functor to launch the groupjoin probe kernel.
 */
struct groupjoin_probe_functor {
  template <typename Identifier>
  void operator()(gqe::context_reference ctx_ref,
                  cudf::column_device_view o_custkey,
                  std::optional<cudf::column_device_view> o_comment,
                  utility::task_hash_map& hash_map_wrapper,
                  rmm::cuda_stream_view stream)
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      // Lazy thread-safe instantiation of CuCo hash map.
      auto& hash_map =
        hash_map_wrapper.get_map<Identifier, cudf::size_type, groupjoin_map_type<Identifier>>();
      auto hash_map_ref = hash_map.ref(cuco::find, cuco::for_each);

      if (o_comment.has_value()) {
        auto grid_size =
          gqe::utility::detect_launch_grid_size(fused_filter_probe_kernel<Identifier>,
                                                utility::block_dim,
                                                /* dynamic_shared_memory_bytes = */ 0);
        fused_filter_probe_kernel<Identifier><<<grid_size, utility::block_dim, 0, stream>>>(
          hash_map_ref, o_custkey, o_comment.value());
      } else {
        auto grid_size =
          gqe::utility::detect_launch_grid_size(groupjoin_probe_kernel<Identifier>,
                                                utility::block_dim,
                                                /* dynamic_shared_memory_bytes = */ 0);
        groupjoin_probe_kernel<Identifier>
          <<<grid_size, utility::block_dim, 0, stream>>>(hash_map_ref, o_custkey);
      }
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

/**
 * @brief Functor to launch the groupjoin retrieve kernel.
 */
struct groupjoin_retrieve_functor {
  template <typename Identifier>
  void operator()(gqe::context_reference ctx_ref,
                  cudf::size_type* d_global_offset,
                  cudf::mutable_column_device_view out_c_custkey,
                  cudf::mutable_column_device_view out_c_count,
                  utility::task_hash_map& hash_map_wrapper,
                  rmm::cuda_stream_view stream)
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      // Lazy thread-safe instantiation of CuCo hash map.
      auto& hash_map =
        hash_map_wrapper.get_map<Identifier, cudf::size_type, groupjoin_map_type<Identifier>>();
      auto hash_map_ref = hash_map.ref(cuco::find, cuco::for_each);

      auto grid_size = gqe::utility::detect_launch_grid_size(groupjoin_retrieve_kernel<Identifier>,
                                                             utility::block_dim,
                                                             /* dynamic_shared_memory_bytes = */ 0);
      groupjoin_retrieve_kernel<Identifier><<<grid_size, utility::block_dim, 0, stream>>>(
        hash_map_ref, d_global_offset, out_c_custkey, out_c_count);
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

groupjoin_probe_task::groupjoin_probe_task(gqe::context_reference ctx_ref,
                                           int32_t task_id,
                                           int32_t stage_id,
                                           std::shared_ptr<gqe::task> groupjoin_build_task,
                                           std::shared_ptr<gqe::task>&& orders_task,
                                           std::shared_ptr<utility::task_hash_map> hash_map,
                                           bool use_fused_kernel)
  : gqe::task(
      ctx_ref, task_id, stage_id, {std::move(groupjoin_build_task), std::move(orders_task)}, {}),
    _hash_map(std::move(hash_map)),
    _use_fused_kernel(use_fused_kernel)
{
}

void groupjoin_probe_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 2);

  auto orders_table    = dependent_tasks[1]->result().value();
  auto const row_count = orders_table.num_rows();

  auto const stream = rmm::cuda_stream_default;

  auto o_custkey_column = cudf::column_device_view::create(orders_table.column(0), stream);

  std::unique_ptr<cudf::column_device_view, std::function<void(cudf::column_device_view*)>>
    o_comment_column =
      _use_fused_kernel ? cudf::column_device_view::create(orders_table.column(1), stream)
                        : nullptr;
  std::optional<cudf::column_device_view> o_comment_view =
    o_comment_column ? std::make_optional(*o_comment_column) : std::nullopt;

  cudf::data_type identifier_type = o_custkey_column->type();

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(identifier_type,
                        groupjoin_probe_functor{},
                        get_context_reference(),
                        *o_custkey_column,
                        o_comment_view,
                        *_hash_map,
                        stream);

  // The probe doesn't emit any results, because the aggregates are complete only after all probe
  // tasks are finished.
  auto empty_table = std::make_unique<cudf::table>();
  emit_result(std::move(empty_table));

  remove_dependencies();
}

std::shared_ptr<utility::task_hash_map> groupjoin_probe_task::hash_map() { return _hash_map; }

groupjoin_retrieve_task::groupjoin_retrieve_task(gqe::context_reference ctx_ref,
                                                 int32_t task_id,
                                                 int32_t stage_id,
                                                 std::shared_ptr<gqe::task> groupjoin_probe_task,
                                                 std::shared_ptr<utility::task_hash_map> hash_map)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(groupjoin_probe_task)}, {}),
    _hash_map(std::move(hash_map))
{
}

void groupjoin_retrieve_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  // In Q13, we know that the cardinality is exact and not an estimate, because `customer` isn't
  // filtered. Thus, we don't need to resize `cudf::column` at the end.
  auto const row_count = _hash_map->cardinality_estimate();

  auto const stream = rmm::cuda_stream_default;

  cudf::data_type identifier_type = _hash_map->identifier_type();
  auto count_type                 = cudf::data_type(cudf::type_to_id<cudf::size_type>());

  auto out_c_custkey_column =
    cudf::make_numeric_column(identifier_type, row_count, cudf::mask_state::UNALLOCATED, stream);
  auto out_c_count_column =
    cudf::make_numeric_column(count_type, row_count, cudf::mask_state::UNALLOCATED, stream);

  auto out_c_custkey_view =
    cudf::mutable_column_device_view::create(out_c_custkey_column->mutable_view(), stream);
  auto out_c_count_view =
    cudf::mutable_column_device_view::create(out_c_count_column->mutable_view(), stream);

  rmm::device_scalar<cudf::size_type> d_global_offset(0, stream);

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(identifier_type,
                        groupjoin_retrieve_functor{},
                        get_context_reference(),
                        d_global_offset.data(),
                        *out_c_custkey_view,
                        *out_c_count_view,
                        *_hash_map,
                        stream);

  GQE_CUDA_TRY(cudaGetLastError());

  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.reserve(2);

  result_columns.push_back(std::move(out_c_custkey_column));
  result_columns.push_back(std::move(out_c_count_column));

  auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
  emit_result(std::move(result_table));

  remove_dependencies();
}

/**
 * @brief Generate the groupjoin build tasks.
 *
 * A build task is generated per build-side input task, so that the build
 * kernels can overlap with build-side transfers.
 *
 * @param[in] input_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
struct groupjoin_build_generate_tasks {
  double scale_factor;

  std::vector<std::shared_ptr<gqe::task>> operator()(
    std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
    gqe::context_reference ctx_ref,
    int32_t& task_id,
    int32_t stage_id)
  {
    assert(input_tasks.size() == 1 && "expected exactly one input relation");

    // The multiplier is taken from the TPC-H definition of the customer table.
    //
    // Refer to TPC-H spec, section 1.4.1 "Required Tables".
    size_t input_cardinality_estimate = 150'000ul * scale_factor;
    auto hash_map = std::make_shared<utility::task_hash_map>(input_cardinality_estimate);

    std::vector<std::shared_ptr<gqe::task>> generated_tasks;
    generated_tasks.reserve(input_tasks[0].size());

    // Generate the probe-side tasks for the orders input.
    for (auto& input_task : input_tasks[0]) {
      generated_tasks.push_back(std::make_shared<groupjoin_build_task>(
        ctx_ref, task_id, stage_id, std::move(input_task), hash_map));
      task_id++;
    }

    return generated_tasks;
  }
};

/**
 * @brief Generate the groupjoin probe tasks.
 *
 * A probe task is generated per probe-side input task, so that the probe
 * kernels can overlap with probe-side transfers.
 *
 * @param[in] input_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
struct groupjoin_probe_generate_tasks {
  bool use_fused_kernel;
  std::vector<std::shared_ptr<gqe::task>> operator()(
    std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
    gqe::context_reference ctx_ref,
    int32_t& task_id,
    int32_t stage_id)
  {
    assert(input_tasks.size() == 2 && "expected a LHS and a RHS input relation");

    // Clone a shared_ptr of the hash map before passing ownership of the LHS input to concatenate
    // task.
    auto build_side_task = dynamic_cast<groupjoin_build_task*>(input_tasks[0][0].get());
    assert(build_side_task != nullptr && "expected a groupjoin build task on the LHS");
    auto hash_map = build_side_task->hash_map();

    std::shared_ptr<gqe::task> concatenated_groupjoin_build;
    // Don't bother concatenating if the number of row groups is only one.
    if (input_tasks[0].size() == 1) {
      concatenated_groupjoin_build = std::move(input_tasks[0][0]);
    } else {
      concatenated_groupjoin_build =
        std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, input_tasks[0]);
      task_id++;
    }

    std::vector<std::shared_ptr<gqe::task>> groupjoin_probe_tasks;
    groupjoin_probe_tasks.reserve(input_tasks[1].size());

    // Generate the probe-side tasks for the orders input.
    for (auto& input_task : input_tasks[1]) {
      groupjoin_probe_tasks.push_back(
        std::make_shared<groupjoin_probe_task>(ctx_ref,
                                               task_id,
                                               stage_id,
                                               concatenated_groupjoin_build,
                                               std::move(input_task),
                                               hash_map,
                                               use_fused_kernel));
      task_id++;
    }

    return groupjoin_probe_tasks;
  }
};

/**
 * @brief Generate the groupjoin retrieve tasks.
 *
 * The retrieve task is generated separately from the probe tasks, so that the
 * probe tasks are scheduled on multiple workers.
 *
 * If we don't do the above, then `groupjoin_probe_generate_tasks` returns a
 * single task, namely the retrieve task, and all dependencies are scheduled on
 * the same worker.
 *
 * @param[in] input_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
struct groupjoin_retrieve_generate_tasks {
  bool use_fused_kernel;
  std::vector<std::shared_ptr<gqe::task>> operator()(
    std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
    gqe::context_reference ctx_ref,
    int32_t& task_id,
    int32_t stage_id)
  {
    assert(input_tasks.size() == 1 && "expected an input relation");

    // Clone a shared_ptr of the hash map before passing ownership of the input
    // to concatenate task.
    auto probe_task = dynamic_cast<groupjoin_probe_task*>(input_tasks[0][0].get());
    assert(probe_task != nullptr && "expected a groupjoin probe task");
    auto hash_map = probe_task->hash_map();

    std::shared_ptr<gqe::task> concatenated_groupjoin_probe;
    // Don't bother concatenating if the number of row groups is only one.
    if (input_tasks[0].size() == 1) {
      concatenated_groupjoin_probe = std::move(input_tasks[0][0]);
    } else {
      concatenated_groupjoin_probe =
        std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, input_tasks[0]);
      task_id++;
    }

    // Convert the hash map into a cuDF table.
    std::shared_ptr<gqe::task> gj_retrieve_task = std::make_shared<groupjoin_retrieve_task>(
      ctx_ref, task_id, stage_id, concatenated_groupjoin_probe, hash_map);
    task_id++;

    return {gj_retrieve_task};
  }
};
}  // namespace

std::shared_ptr<gqe::physical::relation> groupjoin_build(
  std::shared_ptr<gqe::physical::relation> customer, double scale_factor)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {std::move(customer)};

  // TODO: Pass in a size estimate for the hash map. The map is directly built
  // on base table, thus we technically know the size from the GQE catalog.
  // However, although the task graph visitor has access to the catalog, the
  // catalog isn't passed through to the UDR.

  groupjoin_build_generate_tasks build_task_generator{scale_factor};

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, build_task_generator, /* last_child_break_pipeline = */ false);
}

std::shared_ptr<gqe::physical::relation> groupjoin_probe(
  std::shared_ptr<gqe::physical::relation> groupjoin_build,
  std::shared_ptr<gqe::physical::relation> orders)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {std::move(groupjoin_build),
                                                                         std::move(orders)};

  constexpr bool use_fused_kernel = false;
  groupjoin_probe_generate_tasks probe_task_generator{use_fused_kernel};

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, probe_task_generator, /* last_child_break_pipeline = */ false);
}

std::shared_ptr<gqe::physical::relation> fused_filter_probe(
  std::shared_ptr<gqe::physical::relation> groupjoin_build,
  std::shared_ptr<gqe::physical::relation> orders)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {std::move(groupjoin_build),
                                                                         std::move(orders)};

  constexpr bool use_fused_kernel = true;
  groupjoin_probe_generate_tasks probe_task_generator{use_fused_kernel};

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, probe_task_generator, /* last_child_break_pipeline = */ false);
}

std::shared_ptr<gqe::physical::relation> groupjoin_retrieve(
  std::shared_ptr<gqe::physical::relation> groupjoin_probe)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {
    std::move(groupjoin_probe)};

  groupjoin_retrieve_generate_tasks retrieve_task_generator{};

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, retrieve_task_generator, /* last_child_break_pipeline = */ true);
}

}  // namespace q13
}  // namespace benchmark
}  // namespace gqe_python
