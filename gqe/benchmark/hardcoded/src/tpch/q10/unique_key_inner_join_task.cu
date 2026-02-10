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

#include <tpch/q10/unique_key_inner_join.cuh>
#include <tpch/q10/unique_key_inner_join_task.hpp>
#include <utility/config.hpp>
#include <utility/hash_map_cache.hpp>
#include <utility/utility.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/cuda.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace gqe_python {
namespace benchmark {
namespace q10 {

namespace {

template <typename T>
using hash_function = cuco::identity_hash<T>;

template <typename T>
using unique_key_inner_join_map_type = utility::map_type<T, cudf::size_type, hash_function<T>>;

/* Subclasses of gqe::task */

/**
 * @brief A task for the unique-key inner join build.
 */
class unique_key_inner_join_build_task : public gqe::task {
 public:
  /**
   * @brief Construct a unique-key inner join build task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] build_side_table The build-side table.
   * @param[in] hash_map The build-side hash map.
   * @param[in] key_column_idx Column index of the key column.
   * @param[in] enable_bloom_filter Flag for building a Bloom filter to reduce hash map lookup
   * during probing.
   */
  unique_key_inner_join_build_task(gqe::context_reference ctx_ref,
                                   int32_t task_id,
                                   int32_t stage_id,
                                   std::shared_ptr<gqe::task> build_side_table,
                                   std::shared_ptr<utility::task_hash_map> hash_map,
                                   const cudf::size_type key_column_idx,
                                   const bool enable_bloom_filter);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

  std::shared_ptr<utility::task_hash_map> hash_map() { return _hash_map; }

 private:
  std::shared_ptr<utility::task_hash_map> _hash_map;
  cudf::size_type _key_column_idx;
  bool _enable_bloom_filter;
};

/**
 * @brief A task for the unique-key inner join probe.
 */
class unique_key_inner_join_probe_task : public gqe::task {
 public:
  /**
   * @brief Construct a unique-key inner join probe task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] unique_key_inner_join_build_task The input table task.
   * @param[in] build_side_table The build-side table.
   * @param[in] probe_side_table The probe-side table.
   * @param[in] hash_map The hash map built on the key column.
   * @param[in] build_side_key_column_idx Column index of the key column on the build side
   * @param[in] probe_side_key_column_idx Column index of the key column on the probe side
   * @param[in] projection_indices Indices of columns to be included in the output table.
   */
  unique_key_inner_join_probe_task(gqe::context_reference ctx_ref,
                                   int32_t task_id,
                                   int32_t stage_id,
                                   std::shared_ptr<gqe::task> unique_key_inner_join_build_task,
                                   std::shared_ptr<gqe::task> build_side_table,
                                   std::shared_ptr<gqe::task> probe_side_table,
                                   std::shared_ptr<utility::task_hash_map> hash_map,
                                   cudf::size_type build_side_key_column_idx,
                                   cudf::size_type probe_side_key_column_idx,
                                   std::vector<cudf::size_type>& projection_indices);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::shared_ptr<utility::task_hash_map> _hash_map;
  std::vector<cudf::size_type> _projection_indices;
  cudf::size_type _build_side_key_column_idx;
  cudf::size_type _probe_side_key_column_idx;
};

/* Ctors for subclasses of gqe::task */

unique_key_inner_join_build_task::unique_key_inner_join_build_task(
  gqe::context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<gqe::task> build_side_table,
  std::shared_ptr<utility::task_hash_map> hash_map,
  const cudf::size_type key_column_idx,
  const bool enable_bloom_filter)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(build_side_table)}, {}),
    _hash_map(std::move(hash_map)),
    _key_column_idx(key_column_idx),
    _enable_bloom_filter(enable_bloom_filter)
{
}

unique_key_inner_join_probe_task::unique_key_inner_join_probe_task(
  gqe::context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<gqe::task> unique_key_inner_join_build_task,
  std::shared_ptr<gqe::task> build_side_table,
  std::shared_ptr<gqe::task> probe_side_table,
  std::shared_ptr<utility::task_hash_map> hash_map,
  cudf::size_type build_side_key_column_idx,
  cudf::size_type probe_side_key_column_idx,
  std::vector<cudf::size_type>& projection_indices)
  : gqe::task(ctx_ref,
              task_id,
              stage_id,
              {std::move(unique_key_inner_join_build_task),
               std::move(build_side_table),
               std::move(probe_side_table)},
              {}),
    _hash_map(std::move(hash_map)),
    _build_side_key_column_idx(build_side_key_column_idx),
    _probe_side_key_column_idx(probe_side_key_column_idx),
    _projection_indices(projection_indices)
{
}

/* Functor for inserting into hash map and bloom filter, as required by
 * task_hash_map::create_map_bf_and_insert */

template <typename Identifier>
struct insert_functor {
  cudf::column_device_view key_column;
  rmm::cuda_stream_view stream;

  void operator()(
    unique_key_inner_join_map_type<Identifier>& hash_map,
    utility::bloom_filter_type<Identifier, hash_function<Identifier>>& bloom_filter) const
  {
    // Populate hash map
    thrust::for_each(
      thrust::make_counting_iterator<cudf::size_type>(0),
      thrust::make_counting_iterator<cudf::size_type>(key_column.size()),
      [map = hash_map.ref(cuco::insert), keys = key_column] __device__(auto row_idx) mutable {
        // We don't need to check for NULLs here.
        auto key = keys.element<Identifier>(row_idx);
        map.insert(cuco::pair<Identifier, cudf::size_type>(key, row_idx));
      });

    // Populate Bloom filter if initialized with num_bloom_filter_blocks > 0
    if (bloom_filter.block_extent()) {
      auto it =
        thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0),
                                        [keys = key_column] __device__(auto row_idx) -> Identifier {
                                          return keys.element<Identifier>(row_idx);
                                        });
      bloom_filter.add(it, it + key_column.size(), stream);
    }
  }
};

/* Functors to be invoked through dynamic dispatching in `execute()`. */

/**
 * @brief Functor to build the hash map for unique-key inner join.
 */
struct unique_key_inner_join_build_functor {
  template <typename Identifier>
  void operator()(cudf::column_device_view key_column,
                  utility::task_hash_map& hash_map_wrapper,
                  const bool enable_bloom_filter,
                  rmm::cuda_stream_view stream)
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      constexpr int32_t bf_size_factor = 1;
      cuco::extent<std::size_t> num_bloom_filter_blocks =
        enable_bloom_filter ? utility::get_bloom_filter_blocks<
                                utility::bloom_filter_type<Identifier, hash_function<Identifier>>>(
                                bf_size_factor * key_column.size())
                            : cuco::extent<std::size_t>{0};

      // Build hash map and Bloom filter.
      hash_map_wrapper
        .create_map_bf_and_insert<Identifier,
                                  cudf::size_type,
                                  unique_key_inner_join_map_type<Identifier>,
                                  decltype(insert_functor<Identifier>{key_column, stream}),
                                  hash_function<Identifier>>(
          key_column.size(),
          /* load_factor = */ 0.5,
          num_bloom_filter_blocks,
          insert_functor<Identifier>{key_column, stream});
    } else {
      CUDF_FAIL("Key column must be INT32 or INT64");
    }
  }
};

/**
 * @brief Functor to launch the probe kernel for unique-key inner join.
 */
struct unique_key_inner_join_probe_functor {
  template <typename Identifier>
  void operator()(gqe::context_reference ctx_ref,
                  cudf::column_device_view build_side_key_column,
                  cudf::column_device_view probe_side_key_column,
                  cudf::column_device_view l_returnflag,
                  cudf::size_type* d_global_offset,
                  cudf::mutable_column_device_view build_side_indices,
                  cudf::mutable_column_device_view probe_side_indices,
                  utility::task_hash_map& hash_map_wrapper,
                  rmm::cuda_stream_view stream)
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      auto& hash_map =
        hash_map_wrapper
          .get_map<Identifier, cudf::size_type, unique_key_inner_join_map_type<Identifier>>();
      auto hash_map_ref = hash_map.ref(cuco::find, cuco::for_each);
      auto& bloom_filter =
        hash_map_wrapper.get_bloom_filter<Identifier, hash_function<Identifier>>();
      auto bloom_filter_ref = bloom_filter.ref();

      auto grid_size = gqe::utility::detect_launch_grid_size(
        unique_key_inner_join_probe_kernel<Identifier,
                                           decltype(hash_map_ref),
                                           decltype(bloom_filter_ref)>,
        utility::block_dim,
        /* dynamic_shared_memory_bytes = */ 0);
      unique_key_inner_join_probe_kernel<Identifier>
        <<<grid_size, utility::block_dim, 0, stream>>>(hash_map_ref,
                                                       bloom_filter_ref,
                                                       probe_side_key_column,
                                                       l_returnflag,
                                                       d_global_offset,
                                                       {build_side_indices, probe_side_indices});
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

/* Overrides for `execute()` method in subclasses of gqe::task. */

void unique_key_inner_join_build_task::execute()
{
  auto const stream = cudf::get_default_stream();
  prepare_dependencies();
  gqe::utility::nvtx_scoped_range join_task_range("unique_key_join_build_task");
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto input_table = dependent_tasks[0]->result().value();
  auto key_column  = cudf::column_device_view::create(input_table.column(_key_column_idx), stream);

  cudf::data_type identifier_type = key_column->type();

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(identifier_type,
                        unique_key_inner_join_build_functor{},
                        *key_column,
                        *_hash_map,
                        _enable_bloom_filter,
                        stream);

  GQE_CUDA_TRY(cudaGetLastError());
  stream.synchronize();

  auto result_table = std::make_unique<cudf::table>();
  emit_result(std::move(result_table));

  remove_dependencies();
}

void unique_key_inner_join_probe_task::execute()
{
  prepare_dependencies();
  gqe::utility::nvtx_scoped_range join_task_range("unique_key_join_probe_task");
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 3);

  auto build_side_table = dependent_tasks[1]->result().value();
  auto probe_side_table = dependent_tasks[2]->result().value();
  auto const row_count  = probe_side_table.num_rows();

  auto const stream = cudf::get_default_stream();

  auto build_side_key_column =
    cudf::column_device_view::create(build_side_table.column(_build_side_key_column_idx), stream);
  auto probe_side_key_column =
    cudf::column_device_view::create(probe_side_table.column(_probe_side_key_column_idx), stream);
  // hardcoded for Q10
  auto l_returnflag = cudf::column_device_view::create(probe_side_table.column(1), stream);

  auto out_build_side_indices_column =
    cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<cudf::size_type>()),
                              row_count,
                              cudf::mask_state::UNALLOCATED,
                              stream);
  auto out_probe_side_indices_column =
    cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<cudf::size_type>()),
                              row_count,
                              cudf::mask_state::UNALLOCATED,
                              stream);

  auto out_build_side_indices_view =
    cudf::mutable_column_device_view::create(out_build_side_indices_column->mutable_view(), stream);
  auto out_probe_side_indices_view =
    cudf::mutable_column_device_view::create(out_probe_side_indices_column->mutable_view(), stream);

  cudf::data_type identifier_type = probe_side_key_column->type();

  rmm::device_scalar<cudf::size_type> d_global_offset(0, stream);

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(identifier_type,
                        unique_key_inner_join_probe_functor{},
                        get_context_reference(),
                        *build_side_key_column,
                        *probe_side_key_column,
                        *l_returnflag,
                        d_global_offset.data(),
                        *out_build_side_indices_view,
                        *out_probe_side_indices_view,
                        *_hash_map,
                        stream);

  GQE_CUDA_TRY(cudaGetLastError());
  stream.synchronize();

  cudf::size_type h_result_rows = d_global_offset.value(stream);
  gqe::utility::nvtx_mark(std::string("build_size:") + std::to_string(build_side_table.num_rows()) +
                          ", probe_size:" + std::to_string(probe_side_table.num_rows()) +
                          ", result_size:" + std::to_string(h_result_rows));
  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.reserve(_projection_indices.size());

  // separate projection indices into build-side and probe-side for batched gather
  std::vector<cudf::size_type> build_side_projection_indices;
  std::vector<cudf::size_type> probe_side_projection_indices;
  // keep track of indices of projected columns in gathered tables
  // e.g. the i-th projected column is the `gathered_indices[i]`-th element in
  // `build_side_projected_columns` or `probe_side_projected_columns`
  std::vector<cudf::size_type> gathered_indices;
  gathered_indices.reserve(_projection_indices.size());
  for (size_t i = 0; i < _projection_indices.size(); ++i) {
    cudf::size_type projection_idx = _projection_indices[i];
    assert((projection_idx >= 0 &&
            projection_idx < (build_side_table.num_columns() + probe_side_table.num_columns())) &&
           "Projection index out of bounds.");
    if (projection_idx < build_side_table.num_columns()) {
      gathered_indices[i] = build_side_projection_indices.size();
      build_side_projection_indices.push_back(projection_idx);
    } else {
      gathered_indices[i] = probe_side_projection_indices.size();
      projection_idx -= build_side_table.num_columns();
      probe_side_projection_indices.push_back(projection_idx);
    }
  }

  // Lambda for materializing the join output columns
  auto materialize_columns = [](cudf::table_view input_table,
                                std::vector<cudf::size_type>& column_indices,
                                cudf::column_view gather_map) {
    if (column_indices.empty()) { return std::vector<std::unique_ptr<cudf::column>>{}; }
    auto gathered_columns = cudf::gather(input_table.select(column_indices), gather_map)->release();
    return gathered_columns;
  };

  // materialize projected columns from build side
  cudf::column_view build_side_gather_map{
    cudf::data_type(cudf::type_to_id<cudf::size_type>()),
    h_result_rows,
    static_cast<void const*>(out_build_side_indices_view->data<cudf::size_type>()),
    nullptr,
    0};
  auto build_side_projected_columns =
    materialize_columns(build_side_table, build_side_projection_indices, build_side_gather_map);

  // materialize projected columns from probe side
  cudf::column_view probe_side_gather_map{
    cudf::data_type(cudf::type_to_id<cudf::size_type>()),
    h_result_rows,
    static_cast<void const*>(out_probe_side_indices_view->data<cudf::size_type>()),
    nullptr,
    0};
  auto probe_side_projected_columns =
    materialize_columns(probe_side_table, probe_side_projection_indices, probe_side_gather_map);

  // populate result_columns
  for (size_t i = 0; i < _projection_indices.size(); ++i) {
    cudf::size_type projection_idx = _projection_indices[i];
    if (projection_idx < build_side_table.num_columns()) {
      result_columns.push_back(std::move(build_side_projected_columns[gathered_indices[i]]));
    } else {
      result_columns.push_back(std::move(probe_side_projected_columns[gathered_indices[i]]));
    }
  }

  auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
  emit_result(std::move(result_table));

  remove_dependencies();
}

/* Functors for generating tasks. Used by interface functions. */

/**
 * @brief Functor for generating build tasks.
 *
 * A build task is generated per build-side input task, so that the build
 * kernels can overlap with build-side transfers.
 *
 * @param[in] input_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
struct unique_key_inner_join_build_generate_tasks {
  cudf::size_type key_column_idx;
  bool enable_bloom_filter;

  std::vector<std::shared_ptr<gqe::task>> operator()(
    std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
    gqe::context_reference ctx_ref,
    int32_t& task_id,
    int32_t stage_id)
  {
    assert(input_tasks.size() == 1 && "expected exactly one input relation");

    // To specify cardinality during task execution.
    auto hash_map = std::make_shared<utility::task_hash_map>(
      /*cardinality_estimate=*/-1, /*load_factor=*/0.5, enable_bloom_filter);

    std::shared_ptr<gqe::task> build_side_table;
    // Don't bother concatenating if the number of row groups is only one.
    if (input_tasks[0].size() == 1) {
      build_side_table = input_tasks[0][0];
    } else {
      build_side_table =
        std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, input_tasks[0]);
      task_id++;
    }

    return {std::make_shared<unique_key_inner_join_build_task>(
      ctx_ref, task_id, stage_id, build_side_table, hash_map, key_column_idx, enable_bloom_filter)};
  }
};

/**
 * @brief Functor for generating probe tasks.
 *
 * @param[in] input_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
struct unique_key_inner_join_probe_generate_tasks {
  std::vector<cudf::size_type> projection_indices;
  cudf::size_type build_side_key_column_idx;
  cudf::size_type probe_side_key_column_idx;

  std::vector<std::shared_ptr<gqe::task>> operator()(
    std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
    gqe::context_reference ctx_ref,
    int32_t& task_id,
    int32_t stage_id)
  {
    assert(
      input_tasks.size() == 3 &&
      "expected build-side hash map, build-side table and probe-side table as input relations");

    // Clone a shared_ptr of the hash map before passing ownership of the LHS
    // input to concatenate task.
    auto build_side_task = dynamic_cast<unique_key_inner_join_build_task*>(input_tasks[0][0].get());
    assert(build_side_task != nullptr && "expected a unique-key join build task on the LHS");
    auto hash_map = build_side_task->hash_map();

    assert(input_tasks[0].size() == 1 && "expected a single unique-key join build task");
    std::shared_ptr<gqe::task> unique_key_inner_join_build = std::move(input_tasks[0][0]);

    std::shared_ptr<gqe::task> concatenated_unique_key_inner_join_build_side_table;
    // Don't bother concatenating if the number of row groups is only one.
    if (input_tasks[1].size() == 1) {
      concatenated_unique_key_inner_join_build_side_table = std::move(input_tasks[1][0]);
    } else {
      concatenated_unique_key_inner_join_build_side_table =
        std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, input_tasks[1]);
      task_id++;
    }

    std::vector<std::shared_ptr<gqe::task>> unique_key_inner_join_probe_tasks;
    unique_key_inner_join_probe_tasks.reserve(input_tasks[2].size());

    // Generate the probe-side tasks for the orders input.
    for (auto& input_task : input_tasks[2]) {
      unique_key_inner_join_probe_tasks.push_back(
        std::make_shared<unique_key_inner_join_probe_task>(
          ctx_ref,
          task_id,
          stage_id,
          unique_key_inner_join_build,
          concatenated_unique_key_inner_join_build_side_table,
          input_task,
          hash_map,
          build_side_key_column_idx,
          probe_side_key_column_idx,
          projection_indices));
      task_id++;
    }

    return unique_key_inner_join_probe_tasks;
  }
};
}  // namespace

/* Interface function definitions */

std::shared_ptr<gqe::physical::relation> unique_key_inner_join_build(
  std::shared_ptr<gqe::physical::relation> build_side_table,
  const cudf::size_type key_column_idx,
  bool enable_bloom_filter)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {
    std::move(build_side_table)};

  unique_key_inner_join_build_generate_tasks build_task_generator{key_column_idx,
                                                                  enable_bloom_filter};

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, build_task_generator, /* last_child_break_pipeline = */ false);
}

std::shared_ptr<gqe::physical::relation> unique_key_inner_join_probe(
  std::shared_ptr<gqe::physical::relation> build_side_map,
  std::shared_ptr<gqe::physical::relation> build_side_table,
  std::shared_ptr<gqe::physical::relation> probe_side_table,
  cudf::size_type build_side_key_column_idx,
  cudf::size_type probe_side_key_column_idx,
  std::vector<cudf::size_type>& projection_indices)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {
    std::move(build_side_map), std::move(build_side_table), std::move(probe_side_table)};
  unique_key_inner_join_probe_generate_tasks probe_task_generator{
    projection_indices, build_side_key_column_idx, probe_side_key_column_idx};

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, probe_task_generator, /* last_child_break_pipeline = */ false);
}

}  // namespace q10
}  // namespace benchmark
}  // namespace gqe_python
