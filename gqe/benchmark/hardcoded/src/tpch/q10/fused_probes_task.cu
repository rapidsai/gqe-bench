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

#include <tpch/q10/fused_probes.cuh>
#include <tpch/q10/fused_probes_task.hpp>
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

#include <cuda/std/utility>

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
using join_multimap_type = utility::multimap_type<T, cudf::size_type>;

template <typename T>
using join_map_type = utility::map_type<T, cudf::size_type, hash_function<T>>;

/* Subclasses of gqe::task */

/**
 * @brief A task for the join multimap build.
 */
class join_multimap_build_task : public gqe::task {
 public:
  /**
   * @brief Construct a join multimap build task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] build_side_table The build-side table.
   * @param[in] hash_multimap The build-side hash multimap with Bloom filter.
   * @param[in] key_column_idx Index of the key column for building.
   */
  join_multimap_build_task(gqe::context_reference ctx_ref,
                           int32_t task_id,
                           int32_t stage_id,
                           std::shared_ptr<gqe::task> build_side_table,
                           std::shared_ptr<utility::task_hash_map> hash_multimap,
                           const cudf::size_type key_column_idx);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

  std::shared_ptr<utility::task_hash_map> hash_multimap() { return _hash_multimap; }

 private:
  std::shared_ptr<utility::task_hash_map> _hash_multimap;
  cudf::size_type _key_column_idx;
};

/**
 * @brief A task for the join map build.
 */
class join_map_build_task : public gqe::task {
 public:
  /**
   * @brief Construct a unique-key inner join build task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] build_side_table The build-side table.
   * @param[in] hash_map The build-side table.
   * @param[in] key_column_idx Index of the key column for building.
   */
  join_map_build_task(gqe::context_reference ctx_ref,
                      int32_t task_id,
                      int32_t stage_id,
                      std::shared_ptr<gqe::task> build_side_table,
                      std::shared_ptr<utility::task_hash_map> hash_map,
                      const cudf::size_type key_column_idx);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

  std::shared_ptr<utility::task_hash_map> hash_map() { return _hash_map; }

 private:
  std::shared_ptr<utility::task_hash_map> _hash_map;
  cudf::size_type _key_column_idx;
};

/**
 * @brief A task for probing two tables aggregation.
 */
class fused_probes_task : public gqe::task {
 public:
  /**
   * @brief Construct a fused probes and group-by task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] o_custkey_multimap_build The multimap build task on o_custkey.
   * @param[in] n_nationkey_map_build The map build task on n_nationkey.
   * @param[in] join_orders_lineitem_table Table `orders` joined with `lineitem`.
   * @param[in] nation_table The `nation` table.
   * @param[in] customer_table The `customer` table.
   * @param[in] o_custkey_multimap The multimap built on o_custkey.
   * @param[in] n_nationkey_map The map built on n_nationkey.
   */
  fused_probes_task(gqe::context_reference ctx_ref,
                    int32_t task_id,
                    int32_t stage_id,
                    std::shared_ptr<gqe::task> o_custkey_multimap_build,
                    std::shared_ptr<gqe::task> n_nationkey_map_build,
                    std::shared_ptr<gqe::task> join_orders_lineitem_table,
                    std::shared_ptr<gqe::task> nation_table,
                    std::shared_ptr<gqe::task> customer_table,
                    std::shared_ptr<utility::task_hash_map> o_custkey_multimap,
                    std::shared_ptr<utility::task_hash_map> n_nationkey_map);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::shared_ptr<utility::task_hash_map> _o_custkey_multimap;
  std::shared_ptr<utility::task_hash_map> _n_nationkey_map;
};

/* Ctors for subclasses of gqe::task */

join_multimap_build_task::join_multimap_build_task(
  gqe::context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<gqe::task> build_side_table,
  std::shared_ptr<utility::task_hash_map> hash_multimap,
  const cudf::size_type key_column_idx)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(build_side_table)}, {}),
    _hash_multimap(std::move(hash_multimap)),
    _key_column_idx(key_column_idx)
{
}

join_map_build_task::join_map_build_task(gqe::context_reference ctx_ref,
                                         int32_t task_id,
                                         int32_t stage_id,
                                         std::shared_ptr<gqe::task> build_side_table,
                                         std::shared_ptr<utility::task_hash_map> hash_map,
                                         const cudf::size_type key_column_idx)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(build_side_table)}, {}),
    _hash_map(std::move(hash_map)),
    _key_column_idx(key_column_idx)
{
}

fused_probes_task::fused_probes_task(gqe::context_reference ctx_ref,
                                     int32_t task_id,
                                     int32_t stage_id,
                                     std::shared_ptr<gqe::task> o_custkey_multimap_build,
                                     std::shared_ptr<gqe::task> n_nationkey_map_build,
                                     std::shared_ptr<gqe::task> join_orders_lineitem_table,
                                     std::shared_ptr<gqe::task> nation_table,
                                     std::shared_ptr<gqe::task> customer_table,
                                     std::shared_ptr<utility::task_hash_map> o_custkey_multimap,
                                     std::shared_ptr<utility::task_hash_map> n_nationkey_map)
  : gqe::task(ctx_ref,
              task_id,
              stage_id,
              {std::move(o_custkey_multimap_build),
               std::move(n_nationkey_map_build),
               std::move(join_orders_lineitem_table),
               std::move(nation_table),
               std::move(customer_table)},
              {}),
    _o_custkey_multimap(std::move(o_custkey_multimap)),
    _n_nationkey_map(std::move(n_nationkey_map))
{
}

/* Functors for inserting into hash (multi)map and bloom filter, as required by
 * task_hash_map::create_map_bf_and_insert */
/**
 * @brief Functor to insert into hash multimap and Bloom filter.
 */
template <typename Identifier>
struct multimap_bf_insert_functor {
  cudf::column_device_view key_column;
  rmm::cuda_stream_view stream;

  void operator()(
    join_multimap_type<Identifier>& hash_multimap,
    utility::bloom_filter_type<Identifier, hash_function<Identifier>>& bloom_filter) const
  {
    // Populate hash multimap
    thrust::for_each(thrust::make_counting_iterator<cudf::size_type>(0),
                     thrust::make_counting_iterator<cudf::size_type>(key_column.size()),
                     [multimap = hash_multimap.ref(cuco::insert),
                      keys     = key_column] __device__(auto row_idx) mutable {
                       // We don't need to check for NULLs here.
                       auto key = keys.element<Identifier>(row_idx);
                       multimap.insert(cuco::pair<Identifier, cudf::size_type>(key, row_idx));
                     });

    // Populate Bloom filter
    auto it =
      thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0),
                                      [keys = key_column] __device__(auto row_idx) -> Identifier {
                                        return keys.element<Identifier>(row_idx);
                                      });
    bloom_filter.add(it, it + key_column.size(), stream);
  }
};

/**
 * @brief Functor to insert into hash map.
 */
template <typename Identifier>
struct map_insert_functor {
  cudf::column_device_view key_column;
  rmm::cuda_stream_view stream;

  void operator()(join_map_type<Identifier>& hash_map) const
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
  }
};

/* Functors to be invoked through dynamic dispatching in `execute()`. */

struct join_multimap_build_functor {
  template <typename Identifier>
  void operator()(cudf::column_device_view key_column,
                  utility::task_hash_map& hash_multimap_wrapper,
                  rmm::cuda_stream_view stream)
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      constexpr int32_t bf_size_factor                  = 1;
      cuco::extent<std::size_t> num_bloom_filter_blocks = utility::get_bloom_filter_blocks<
        utility::bloom_filter_type<Identifier, hash_function<Identifier>>>(bf_size_factor *
                                                                           key_column.size());

      // Build hash multimap and Bloom filter.
      hash_multimap_wrapper.create_map_bf_and_insert<
        Identifier,
        cudf::size_type,
        join_multimap_type<Identifier>,
        decltype(multimap_bf_insert_functor<Identifier>{key_column, stream}),
        hash_function<Identifier>>(key_column.size(),
                                   /* load_factor = */ 0.5,
                                   num_bloom_filter_blocks,
                                   multimap_bf_insert_functor<Identifier>{key_column, stream});
    } else {
      CUDF_FAIL("Key column must be INT32 or INT64");
    }
  }
};

/**
 * @brief Functor to build the hash map for join.
 */
struct join_map_build_functor {
  template <typename Identifier>
  void operator()(cudf::column_device_view key_column,
                  utility::task_hash_map& hash_map_wrapper,
                  rmm::cuda_stream_view stream)
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      // Build hash map.
      hash_map_wrapper
        .create_map_and_insert<Identifier, cudf::size_type, join_map_type<Identifier>>(
          key_column.size(),
          /* load_factor = */ 0.5,
          map_insert_functor<Identifier>{key_column, stream});
    } else {
      CUDF_FAIL("Key column must be INT32 or INT64");
    }
  }
};

/**
 * @brief Functor to launch the fused probes kernel for join.
 */
struct fused_probes_functor {
  template <typename Identifier>
  void operator()(gqe::context_reference ctx_ref,
                  cudf::column_device_view c_custkey_column,
                  cudf::column_device_view c_nationkey_column,
                  cudf::size_type* d_global_offset,
                  cudf::mutable_column_device_view join_orders_lineitem_table_indices,
                  cudf::mutable_column_device_view nation_table_indices,
                  cudf::mutable_column_device_view customer_table_indices,
                  utility::task_hash_map& o_custkey_multimap_wrapper,
                  utility::task_hash_map& n_nationkey_map_wrapper,
                  rmm::cuda_stream_view stream)
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      auto& o_custkey_multimap =
        o_custkey_multimap_wrapper
          .get_map<Identifier, cudf::size_type, join_multimap_type<Identifier>>();
      auto& o_custkey_bloom_filter =
        o_custkey_multimap_wrapper.get_bloom_filter<Identifier, hash_function<Identifier>>();
      auto& n_nationkey_map =
        n_nationkey_map_wrapper.get_map<Identifier, cudf::size_type, join_map_type<Identifier>>();
      auto o_custkey_multimap_ref     = o_custkey_multimap.ref(cuco::find, cuco::for_each);
      auto o_custkey_bloom_filter_ref = o_custkey_bloom_filter.ref();
      auto n_nationkey_map_ref        = n_nationkey_map.ref(cuco::find, cuco::for_each);

      auto grid_size = gqe::utility::detect_launch_grid_size(
        fused_probes_kernel<Identifier,
                            decltype(o_custkey_multimap_ref),
                            decltype(o_custkey_bloom_filter_ref),
                            decltype(n_nationkey_map_ref)>,
        utility::block_dim,
        /* dynamic_shared_memory_bytes = */ 0);
      fused_probes_kernel<Identifier><<<grid_size, utility::block_dim, 0, stream>>>(
        o_custkey_multimap_ref,
        o_custkey_bloom_filter_ref,
        n_nationkey_map_ref,
        c_custkey_column,
        c_nationkey_column,
        d_global_offset,
        {join_orders_lineitem_table_indices, nation_table_indices, customer_table_indices});
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

/* Overrides for `execute()` method in subclasses of gqe::task. */

void join_multimap_build_task::execute()
{
  auto const stream = cudf::get_default_stream();
  prepare_dependencies();
  gqe::utility::nvtx_scoped_range task_range("join_multimap_build_task");
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto input_table = dependent_tasks[0]->result().value();
  auto key_column  = cudf::column_device_view::create(input_table.column(_key_column_idx), stream);

  cudf::data_type identifier_type = key_column->type();

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(
    identifier_type, join_multimap_build_functor{}, *key_column, *_hash_multimap, stream);

  GQE_CUDA_TRY(cudaGetLastError());
  stream.synchronize();

  auto result_table = std::make_unique<cudf::table>();
  emit_result(std::move(result_table));

  remove_dependencies();
}

void join_map_build_task::execute()
{
  auto const stream = cudf::get_default_stream();
  prepare_dependencies();
  gqe::utility::nvtx_scoped_range join_task_range("join_map_build_task");
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto input_table = dependent_tasks[0]->result().value();
  auto key_column  = cudf::column_device_view::create(input_table.column(_key_column_idx), stream);

  cudf::data_type identifier_type = key_column->type();

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(identifier_type, join_map_build_functor{}, *key_column, *_hash_map, stream);

  GQE_CUDA_TRY(cudaGetLastError());
  stream.synchronize();

  auto result_table = std::make_unique<cudf::table>();
  emit_result(std::move(result_table));

  remove_dependencies();
}

void fused_probes_task::execute()
{
  prepare_dependencies();
  gqe::utility::nvtx_scoped_range join_task_range("fused_probes_task");
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 5);

  auto join_orders_lineitem_table = dependent_tasks[2]->result().value();
  auto nation_table               = dependent_tasks[3]->result().value();
  auto customer_table             = dependent_tasks[4]->result().value();
  // `join_orders_lineitem_table` is joined with `customer` on o_custkey = c_custkey. Since
  // c_custkey is distinct, each row in o_custkey can at most lead to one output row. Note that this
  // argument is independent of which column the current implementation uses for probing.
  auto const output_size_upper_bound = join_orders_lineitem_table.num_rows();

  auto const stream = cudf::get_default_stream();

  auto c_custkey_column   = cudf::column_device_view::create(customer_table.column(0), stream);
  auto c_nationkey_column = cudf::column_device_view::create(customer_table.column(1), stream);

  auto out_join_orders_lineitem_table_indices_column =
    cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<cudf::size_type>()),
                              output_size_upper_bound,
                              cudf::mask_state::UNALLOCATED,
                              stream);
  auto out_nation_table_indices_column =
    cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<cudf::size_type>()),
                              output_size_upper_bound,
                              cudf::mask_state::UNALLOCATED,
                              stream);
  auto out_customer_table_indices_column =
    cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<cudf::size_type>()),
                              output_size_upper_bound,
                              cudf::mask_state::UNALLOCATED,
                              stream);

  auto out_join_orders_lineitem_table_indices_view = cudf::mutable_column_device_view::create(
    out_join_orders_lineitem_table_indices_column->mutable_view(), stream);
  auto out_nation_table_indices_view = cudf::mutable_column_device_view::create(
    out_nation_table_indices_column->mutable_view(), stream);
  auto out_customer_table_indices_view = cudf::mutable_column_device_view::create(
    out_customer_table_indices_column->mutable_view(), stream);

  cudf::data_type identifier_type = c_custkey_column->type();

  rmm::device_scalar<cudf::size_type> d_global_offset(0, stream);

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(identifier_type,
                        fused_probes_functor{},
                        get_context_reference(),
                        *c_custkey_column,
                        *c_nationkey_column,
                        d_global_offset.data(),
                        *out_join_orders_lineitem_table_indices_view,
                        *out_nation_table_indices_view,
                        *out_customer_table_indices_view,
                        *_o_custkey_multimap,
                        *_n_nationkey_map,
                        stream);
  GQE_CUDA_TRY(cudaGetLastError());
  stream.synchronize();

  cudf::size_type h_result_rows = d_global_offset.value(stream);
  gqe::utility::nvtx_mark(std::string("join_orders_lineitem_table size:") +
                          std::to_string(join_orders_lineitem_table.num_rows()) +
                          ", nation_table size:" + std::to_string(nation_table.num_rows()) +
                          ", customer_table size:" + std::to_string(customer_table.num_rows()) +
                          ", result_size:" + std::to_string(h_result_rows));

  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.reserve(9);

  // Lambda for materializing the join output columns
  auto materialize_columns = [](cudf::table_view input_table,
                                std::vector<cudf::size_type>& column_indices,
                                cudf::column_view gather_map) {
    if (column_indices.empty()) { return std::vector<std::unique_ptr<cudf::column>>{}; }
    auto gathered_columns = cudf::gather(input_table.select(column_indices), gather_map)->release();
    return gathered_columns;
  };

  // materialize output columns from joined orders and lineitem table: l_extendedprice, l_discount
  cudf::column_view join_orders_lineitem_table_gather_map{
    cudf::data_type(cudf::type_to_id<cudf::size_type>()),
    h_result_rows,
    static_cast<void const*>(out_join_orders_lineitem_table_indices_view->data<cudf::size_type>()),
    nullptr,
    0};
  std::vector<cudf::size_type> join_orders_lineitem_table_column_indices = {1, 2};
  auto out_join_orders_lineitem_table_columns =
    materialize_columns(join_orders_lineitem_table,
                        join_orders_lineitem_table_column_indices,
                        join_orders_lineitem_table_gather_map);

  // materialize output columns from nation table: n_name
  cudf::column_view nation_table_gather_map{
    cudf::data_type(cudf::type_to_id<cudf::size_type>()),
    h_result_rows,
    static_cast<void const*>(out_nation_table_indices_view->data<cudf::size_type>()),
    nullptr,
    0};
  std::vector<cudf::size_type> nation_table_column_indices = {1};
  auto out_nation_table_columns =
    materialize_columns(nation_table, nation_table_column_indices, nation_table_gather_map);

  // materialize output columns from customer table: c_custkey, c_name, c_acctbal, c_phone,
  // c_address, c_comment
  cudf::column_view customer_table_gather_map{
    cudf::data_type(cudf::type_to_id<cudf::size_type>()),
    h_result_rows,
    static_cast<void const*>(out_customer_table_indices_view->data<cudf::size_type>()),
    nullptr,
    0};
  std::vector<cudf::size_type> customer_table_column_indices = {0, 2, 3, 4, 5, 6};
  auto out_customer_table_columns =
    materialize_columns(customer_table, customer_table_column_indices, customer_table_gather_map);

  // populate result_columns
  result_columns.insert(result_columns.end(),
                        std::make_move_iterator(out_join_orders_lineitem_table_columns.begin()),
                        std::make_move_iterator(out_join_orders_lineitem_table_columns.end()));
  result_columns.insert(result_columns.end(),
                        std::make_move_iterator(out_nation_table_columns.begin()),
                        std::make_move_iterator(out_nation_table_columns.end()));
  result_columns.insert(result_columns.end(),
                        std::make_move_iterator(out_customer_table_columns.begin()),
                        std::make_move_iterator(out_customer_table_columns.end()));

  auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
  emit_result(std::move(result_table));

  remove_dependencies();
}

/* Functors for generating tasks. Used by interface functions. */

/**
 * @brief Functor for generating multimap build tasks.
 *
 * A build task is generated per build-side input task, so that the build
 * kernels can overlap with build-side transfers.
 *
 * @param[in] input_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
struct join_multimap_build_generate_tasks {
  cudf::size_type key_column_idx;

  std::vector<std::shared_ptr<gqe::task>> operator()(
    std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
    gqe::context_reference ctx_ref,
    int32_t& task_id,
    int32_t stage_id)
  {
    assert(input_tasks.size() == 1 && "expected exactly one input relation");

    // To specify cardinality during task execution.
    auto hash_multimap = std::make_shared<utility::task_hash_map>(
      /*cardinality_estimate=*/-1, /*load_factor=*/0.5, /*enable_bloom_filter=*/true);

    std::shared_ptr<gqe::task> build_side_table;
    // Don't bother concatenating if the number of row groups is only one.
    if (input_tasks[0].size() == 1) {
      build_side_table = input_tasks[0][0];
    } else {
      build_side_table =
        std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, input_tasks[0]);
      task_id++;
    }

    return {std::make_shared<join_multimap_build_task>(
      ctx_ref, task_id, stage_id, build_side_table, hash_multimap, key_column_idx)};
  }
};

/**
 * @brief Functor for generating map build tasks.
 *
 * @param[in] input_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
struct join_map_build_generate_tasks {
  cudf::size_type key_column_idx;

  std::vector<std::shared_ptr<gqe::task>> operator()(
    std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
    gqe::context_reference ctx_ref,
    int32_t& task_id,
    int32_t stage_id)
  {
    assert(input_tasks.size() == 1 && "expected exactly one input relation");

    // To specify cardinality during task execution.
    auto hash_map = std::make_shared<utility::task_hash_map>(
      /*cardinality_estimate=*/-1, /*load_factor=*/0.5, /*enable_bloom_filter=*/false);

    std::shared_ptr<gqe::task> build_side_table;
    // Don't bother concatenating if the number of row groups is only one.
    if (input_tasks[0].size() == 1) {
      build_side_table = input_tasks[0][0];
    } else {
      build_side_table =
        std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, input_tasks[0]);
      task_id++;
    }

    return {std::make_shared<join_map_build_task>(
      ctx_ref, task_id, stage_id, build_side_table, hash_map, key_column_idx)};
  }
};

/**
 * @brief Functor for generating fused probes tasks.
 *
 * @param[in] input_tasks The input tasks.
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 */
struct fused_probes_generate_tasks {
  std::vector<std::shared_ptr<gqe::task>> operator()(
    std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
    gqe::context_reference ctx_ref,
    int32_t& task_id,
    int32_t stage_id)
  {
    assert(
      input_tasks.size() == 5 &&
      "expected 5 input relations: 1 hash multimap and 1 hash map for probing, and 3 input tables");

    // Clone shared_ptr of the hash (multi)maps before passing ownership of the
    // input to concatenate task.
    auto o_custkey_multimap_build =
      dynamic_cast<join_multimap_build_task*>(input_tasks[0][0].get());
    assert(o_custkey_multimap_build != nullptr && "expected a join multimap build task");
    auto o_custkey_multimap = o_custkey_multimap_build->hash_multimap();

    auto n_nationkey_map_build = dynamic_cast<join_map_build_task*>(input_tasks[1][0].get());
    assert(n_nationkey_map_build != nullptr && "expected a unique-key inner join build task");
    auto n_nationkey_map = n_nationkey_map_build->hash_map();

    assert(input_tasks[0].size() == 1 &&
           "expected a single hash multimap build task for o_custkey");
    std::shared_ptr<gqe::task> o_custkey_multimap_build_task = std::move(input_tasks[0][0]);

    assert(input_tasks[1].size() == 1 && "expected a single hash map build task for n_nationkey");
    std::shared_ptr<gqe::task> n_nationkey_map_build_task = std::move(input_tasks[1][0]);

    std::shared_ptr<gqe::task> concatenated_join_orders_lineitem_table;
    // Don't bother concatenating if the number of row groups is only one.
    if (input_tasks[2].size() == 1) {
      concatenated_join_orders_lineitem_table = std::move(input_tasks[2][0]);
    } else {
      concatenated_join_orders_lineitem_table =
        std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, input_tasks[2]);
      task_id++;
    }

    std::shared_ptr<gqe::task> concatenated_nation_table;
    // Don't bother concatenating if the number of row groups is only one.
    if (input_tasks[3].size() == 1) {
      concatenated_nation_table = std::move(input_tasks[3][0]);
    } else {
      concatenated_nation_table =
        std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, input_tasks[3]);
      task_id++;
    }

    std::vector<std::shared_ptr<gqe::task>> fused_probes_tasks;
    fused_probes_tasks.reserve(input_tasks[4].size());

    // Generate the probe-side tasks for the customer input.
    for (auto& input_task : input_tasks[4]) {
      fused_probes_tasks.push_back(
        std::make_shared<fused_probes_task>(ctx_ref,
                                            task_id,
                                            stage_id,
                                            o_custkey_multimap_build_task,
                                            n_nationkey_map_build_task,
                                            concatenated_join_orders_lineitem_table,
                                            concatenated_nation_table,
                                            input_task,
                                            o_custkey_multimap,
                                            n_nationkey_map));
      task_id++;
    }

    return fused_probes_tasks;
  }
};

}  // namespace

/* Interface function definitions */

std::shared_ptr<gqe::physical::relation> fused_probes_join_multimap_build(
  std::shared_ptr<gqe::physical::relation> build_side_table, const cudf::size_type key_column_idx)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {
    std::move(build_side_table)};

  join_multimap_build_generate_tasks build_task_generator{key_column_idx};

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, build_task_generator, /* last_child_break_pipeline = */ false);
}

std::shared_ptr<gqe::physical::relation> fused_probes_join_map_build(
  std::shared_ptr<gqe::physical::relation> build_side_table, const cudf::size_type key_column_idx)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {
    std::move(build_side_table)};

  join_map_build_generate_tasks build_task_generator{key_column_idx};

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, build_task_generator, /* last_child_break_pipeline = */ false);
}

std::shared_ptr<gqe::physical::relation> fused_probes_join_probe(
  std::shared_ptr<gqe::physical::relation> o_custkey_to_row_indices_multimap,
  std::shared_ptr<gqe::physical::relation> n_nationkey_to_row_index_map,
  std::shared_ptr<gqe::physical::relation> join_orders_lineitem_table,
  std::shared_ptr<gqe::physical::relation> nation_table,
  std::shared_ptr<gqe::physical::relation> customer_table)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {
    std::move(o_custkey_to_row_indices_multimap),
    std::move(n_nationkey_to_row_index_map),
    std::move(join_orders_lineitem_table),
    std::move(nation_table),
    std::move(customer_table)};

  fused_probes_generate_tasks fused_probes_task_generator;

  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, fused_probes_task_generator, /* last_child_break_pipeline = */ false);
}

}  // namespace q10
}  // namespace benchmark
}  // namespace gqe_python
