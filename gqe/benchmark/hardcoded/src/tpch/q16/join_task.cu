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

#include <tpch/q16/fused_filter_join.cuh>
#include <tpch/q16/task.hpp>
#include <utility/hash_map_cache.hpp>
#include <utility/utility.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/executor/aggregate.hpp>
#include <gqe/executor/groupby.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/cuda.hpp>

namespace gqe_python {
namespace benchmark {
namespace q16 {

template <typename T>
using part_map_type = utility::map_type<T, cudf::size_type>;

template <typename T>
using supplier_map_type = utility::map_type<T, cudf::size_type>;

// Here disable the bloom filter for performance regression.
static constexpr bool enable_part_bloom_filter     = false;
static constexpr bool enable_supplier_bloom_filter = false;

/**
 * @brief A customized task to have a customized fused kernel for filter and hash join.
 */
class fused_filter_join_task : public gqe::task {
 public:
  /**
   * @brief Construct a customized fused task of filter and hash join.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] supplier_table Supplier table as input to join tasks.
   * @param[in] part_table Part table as input to join tasks.
   * @param[in] partsupp_table Partsupp table as input to join tasks.
   * @param[in] supplier_hash_map The hash map built from the supplier table.
   * @param[in] part_hash_map The hash map built from the part table.
   */
  fused_filter_join_task(gqe::context_reference ctx_ref,
                         int32_t task_id,
                         int32_t stage_id,
                         std::shared_ptr<gqe::task> supplier_table,
                         std::shared_ptr<gqe::task> part_table,
                         std::shared_ptr<gqe::task> partsupp_table,
                         std::shared_ptr<utility::task_hash_map> supplier_hash_map,
                         std::shared_ptr<utility::task_hash_map> part_hash_map);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

  /**
   * @brief Return the supplier hash map.
   */
  std::shared_ptr<utility::task_hash_map> supplier_hash_map();

  /**
   * @brief Return the part hash map.
   */
  std::shared_ptr<utility::task_hash_map> part_hash_map();

 private:
  std::shared_ptr<utility::task_hash_map> _supplier_hash_map;
  std::shared_ptr<utility::task_hash_map> _part_hash_map;
};

fused_filter_join_task::fused_filter_join_task(
  gqe::context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<gqe::task> supplier_table,
  std::shared_ptr<gqe::task> part_table,
  std::shared_ptr<gqe::task> partsupp_table,
  std::shared_ptr<utility::task_hash_map> supplier_hash_map,
  std::shared_ptr<utility::task_hash_map> part_hash_map)
  : gqe::task(ctx_ref,
              task_id,
              stage_id,
              {std::move(supplier_table), std::move(part_table), std::move(partsupp_table)},
              {}),
    _supplier_hash_map(std::move(supplier_hash_map)),
    _part_hash_map(std::move(part_hash_map))
{
}

std::shared_ptr<utility::task_hash_map> fused_filter_join_task::supplier_hash_map()
{
  return _supplier_hash_map;
}

std::shared_ptr<utility::task_hash_map> fused_filter_join_task::part_hash_map()
{
  return _part_hash_map;
}

/**
 * @brief Functor to insert into the part hash map.
 */
template <typename Identifier>
struct part_insertion_functor {
  gqe::context_reference ctx_ref;
  cudf::column_device_view partkey_column;
  cudf::column_device_view brand_column;
  cudf::column_device_view type_column;
  cudf::column_device_view size_column;
  rmm::cuda_stream_view main_stream;

  void operator()(part_map_type<Identifier>& part_map,
                  utility::bloom_filter_type<Identifier>& part_bloom_filter) const
  {
    auto part_map_ref          = part_map.ref(cuco::op::find, cuco::op::for_each, cuco::op::insert);
    auto part_bloom_filter_ref = part_bloom_filter.ref();

    auto const grid_size = gqe::utility::detect_launch_grid_size(
      fused_filter_hashtable_build_kernel<Identifier, enable_part_bloom_filter>,
      utility::block_dim,
      /* dynamic_shared_memory_bytes = */ 0);

    fused_filter_hashtable_build_kernel<Identifier, enable_part_bloom_filter>
      <<<grid_size, gqe_python::utility::block_dim, 0, main_stream>>>(partkey_column,
                                                                      brand_column,
                                                                      type_column,
                                                                      size_column,
                                                                      part_map_ref,
                                                                      part_bloom_filter_ref);
  }
};

/**
 * @brief Functor to insert into the supplier hash map.
 */
template <typename Identifier>
struct supplier_insertion_functor {
  cudf::size_type s_suppkey_size;
  cudf::column_device_view suppkey_column;
  void operator()(supplier_map_type<Identifier>& supplier_map,
                  utility::bloom_filter_type<Identifier>& supplier_bloom_filter) const
  {
    auto supplier_map_ref = supplier_map.ref(cuco::op::find, cuco::op::for_each, cuco::op::insert);
    auto supplier_bloom_filter_ref = supplier_bloom_filter.ref();

    thrust::for_each(thrust::make_counting_iterator<cudf::size_type>(0),
                     thrust::make_counting_iterator<cudf::size_type>(s_suppkey_size),
                     [supplier_map_ref,
                      supplier_bloom_filter_ref,
                      keys = suppkey_column] __device__(auto row_idx) mutable {
                       supplier_map_ref.insert(thrust::pair<Identifier, Identifier>(
                         keys.element<Identifier>(row_idx), row_idx));
                       if constexpr (enable_supplier_bloom_filter) {
                         supplier_bloom_filter_ref.add(keys.element<Identifier>(row_idx));
                       }
                     });
  }
};

/**
 * @brief Functor for building the hash map for supplier table and part table.
 */
struct fused_filter_join_build_functor {
  gqe::context_reference ctx_ref;
  cudf::table_view supplier_table;
  cudf::table_view part_table;
  rmm::cuda_stream_view main_stream;
  utility::task_hash_map& part_hash_map_wrapper;
  utility::task_hash_map& supplier_hash_map_wrapper;

  template <typename Identifier>
  void operator()() const
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      auto const s_row_count = supplier_table.num_rows();
      auto const p_row_count = part_table.num_rows();
      // Load the necessary columns.
      auto s_suppkey_column =
        cudf::column_device_view::create(supplier_table.column(0), main_stream);
      auto p_partkey_column = cudf::column_device_view::create(part_table.column(0), main_stream);
      auto p_brand_column   = cudf::column_device_view::create(part_table.column(1), main_stream);
      auto p_type_column    = cudf::column_device_view::create(part_table.column(2), main_stream);
      auto p_size_column    = cudf::column_device_view::create(part_table.column(3), main_stream);

      // There are two approaches to get the estimated cardinality of the hash map.
      // 1) Use an estimated join selectivity.
      // 2) Get the boolean mask column and run reduce to get the exact cardinality.
      // Here we adopt the first approach for better performance and simplicity.
      // But second approach would not degrade the performance too much.

      // 0.16 is the estimated selectivity for part and partsupp join from umbra.
      auto constexpr part_selectivity                  = 0.16;
      auto constexpr part_load_factor                  = 0.5;
      auto constexpr part_bf_size_factor               = 2 * part_selectivity;
      cuco::extent<std::size_t> part_num_filter_blocks = 0;
      if constexpr (enable_part_bloom_filter) {
        part_num_filter_blocks =
          utility::get_bloom_filter_blocks<gqe_python::utility::bloom_filter_type<Identifier>>(
            part_bf_size_factor * p_row_count);
      }

      // Would implicitly build the hash map.
      part_hash_map_wrapper
        .create_map_bf_and_insert<Identifier, cudf::size_type, part_map_type<Identifier>>(
          p_row_count * part_selectivity,
          part_load_factor,
          part_num_filter_blocks,
          part_insertion_functor<Identifier>{ctx_ref,
                                             *p_partkey_column,
                                             *p_brand_column,
                                             *p_type_column,
                                             *p_size_column,
                                             main_stream});

      // Build supplier hash table.
      auto constexpr supplier_load_factor                  = 0.5;
      auto constexpr supplier_bf_size_factor               = 2;
      cuco::extent<std::size_t> supplier_num_filter_blocks = 0;
      if constexpr (enable_supplier_bloom_filter) {
        supplier_num_filter_blocks =
          utility::get_bloom_filter_blocks<gqe_python::utility::bloom_filter_type<Identifier>>(
            supplier_bf_size_factor * s_row_count);
      }

      supplier_hash_map_wrapper
        .create_map_bf_and_insert<Identifier, cudf::size_type, supplier_map_type<Identifier>>(
          s_row_count,
          supplier_load_factor,
          supplier_num_filter_blocks,
          supplier_insertion_functor<Identifier>{s_row_count, *s_suppkey_column});
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

/**
 * @brief Functor to perform fused filter and join.
 *
 */
struct fused_filter_join_probe_functor {
  gqe::context_reference ctx_ref;
  cudf::table_view supplier_table;
  cudf::table_view part_table;
  cudf::table_view part_supp_table;
  rmm::cuda_stream_view main_stream;
  cudf::size_type* d_counter_ptr;
  cudf::size_type* p_out_indices;
  cudf::size_type* ps_out_indices;
  utility::task_hash_map& part_hash_map_wrapper;
  utility::task_hash_map& supplier_hash_map_wrapper;

  template <typename Identifier>
  void operator()() const
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      // Load the necessary columns.
      auto ps_partkey_column =
        cudf::column_device_view::create(part_supp_table.column(0), main_stream);
      auto ps_suppkey_column =
        cudf::column_device_view::create(part_supp_table.column(1), main_stream);

      auto& part_map =
        part_hash_map_wrapper.get_map<Identifier, cudf::size_type, part_map_type<Identifier>>();
      auto& part_bloom_filter = part_hash_map_wrapper.get_bloom_filter<Identifier>();
      auto part_map_ref       = part_map.ref(cuco::op::find, cuco::op::for_each, cuco::op::insert);
      auto& supplier_map      = supplier_hash_map_wrapper
                             .get_map<Identifier, cudf::size_type, supplier_map_type<Identifier>>();
      auto& supplier_bloom_filter = supplier_hash_map_wrapper.get_bloom_filter<Identifier>();
      auto supplier_map_ref =
        supplier_map.ref(cuco::op::find, cuco::op::for_each, cuco::op::insert);
      auto part_bloom_filter_ref     = part_bloom_filter.ref();
      auto supplier_bloom_filter_ref = supplier_bloom_filter.ref();

      // Fused hash join.
      fused_join_params<Identifier> params{*ps_partkey_column,
                                           *ps_suppkey_column,
                                           supplier_map_ref,
                                           supplier_bloom_filter_ref,
                                           part_map_ref,
                                           part_bloom_filter_ref,
                                           p_out_indices,
                                           ps_out_indices,
                                           d_counter_ptr};

      auto const grid_size = gqe::utility::detect_launch_grid_size(
        fused_join_kernel<Identifier, enable_part_bloom_filter, enable_supplier_bloom_filter>,
        utility::block_dim,
        /* dynamic_shared_memory_bytes = */ 0);

      fused_join_kernel<Identifier, enable_part_bloom_filter, enable_supplier_bloom_filter>
        <<<grid_size, gqe_python::utility::block_dim, 0, main_stream>>>(params);
    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

void fused_filter_join_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 3 && "fused_filter_join expected exactly three input relations");

  auto const main_stream = cudf::get_default_stream();
  gqe::utility::nvtx_scoped_range fused_filter_join_task_range("q16_fused_filter_join_task");

  // The supplier table is after the first filter.
  auto supplier_table  = dependent_tasks[0]->result().value();
  auto part_table      = dependent_tasks[1]->result().value();
  auto part_supp_table = dependent_tasks[2]->result().value();

  cudf::data_type identifier_type = supplier_table.column(0).type();
  auto const ps_row_count         = part_supp_table.num_rows();

  // Build the hash maps for supplier and part tables.
  cudf::type_dispatcher(identifier_type,
                        fused_filter_join_build_functor{get_context_reference(),
                                                        supplier_table,
                                                        part_table,
                                                        main_stream,
                                                        *_part_hash_map,
                                                        *_supplier_hash_map});
  GQE_CUDA_TRY(cudaGetLastError());

  rmm::device_uvector<cudf::size_type> p_out_indices(ps_row_count, main_stream);
  rmm::device_uvector<cudf::size_type> ps_out_indices(ps_row_count, main_stream);
  rmm::device_scalar<cudf::size_type> d_counter(0, main_stream);

  // Probe the hash maps.
  cudf::type_dispatcher(identifier_type,
                        fused_filter_join_probe_functor{get_context_reference(),
                                                        supplier_table,
                                                        part_table,
                                                        part_supp_table,
                                                        main_stream,
                                                        d_counter.data(),
                                                        p_out_indices.data(),
                                                        ps_out_indices.data(),
                                                        *_part_hash_map,
                                                        *_supplier_hash_map});

  main_stream.synchronize();
  GQE_CUDA_TRY(cudaGetLastError());
  cudf::size_type h_counter = d_counter.value(main_stream);

  // Materialize the output results.
  auto materialize_column =
    [](cudf::table_view input_table, cudf::size_type column_idx, cudf::column_view gather_map) {
      auto gathered_column = cudf::gather(input_table.select({column_idx}), gather_map)->release();
      return std::move(gathered_column[0]);
    };
  cudf::column_view p_gather_map{cudf::data_type{cudf::type_id::INT32},
                                 static_cast<cudf::size_type>(h_counter),
                                 static_cast<void const*>(p_out_indices.data()),
                                 nullptr,
                                 0};
  cudf::column_view ps_gather_map{cudf::data_type{cudf::type_id::INT32},
                                  static_cast<cudf::size_type>(h_counter),
                                  static_cast<void const*>(ps_out_indices.data()),
                                  nullptr,
                                  0};
  std::vector<std::unique_ptr<cudf::column>> out_columns;
  out_columns.push_back(materialize_column(part_table, 1, p_gather_map));
  out_columns.push_back(materialize_column(part_table, 2, p_gather_map));
  out_columns.push_back(materialize_column(part_table, 3, p_gather_map));
  out_columns.push_back(materialize_column(part_supp_table, 1, ps_gather_map));
  emit_result(std::make_unique<cudf::table>(std::move(out_columns)));
  remove_dependencies();
}

/**
 * @brief Generate the tasks for fused filter and join relation. We concatenate the build side input
 * tables if there are more than 1 partition.
 *
 * @param[in] children_tasks children tasks
 * @param[in] ctx_ref context reference
 * @param[in] task_id task ID
 * @param[in] stage_id stage ID
 */
std::vector<std::shared_ptr<gqe::task>> fused_filter_join_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id)
{
  assert(children_tasks.size() == 3 &&
         "fused_filter_join_probe expected exactly three input relations");

  std::shared_ptr<gqe::task> supplier_table =
    utility::concatenate_tasks(ctx_ref, task_id, stage_id, children_tasks[0]);
  std::shared_ptr<gqe::task> part_table =
    utility::concatenate_tasks(ctx_ref, task_id, stage_id, children_tasks[1]);

  // Would provide the estimate cardinality during task execution.
  auto supplier_hash_map = std::make_shared<utility::task_hash_map>(
    /*cardinality_estimate=*/-1, /*load_factor=*/0.5, enable_supplier_bloom_filter);
  auto part_hash_map = std::make_shared<utility::task_hash_map>(
    /*cardinality_estimate=*/-1, /*load_factor=*/0.5, enable_part_bloom_filter);

  std::vector<std::shared_ptr<gqe::task>> pipeline_results;
  for (auto& partsupp_task : children_tasks[2]) {
    pipeline_results.push_back(std::make_shared<fused_filter_join_task>(ctx_ref,
                                                                        task_id,
                                                                        stage_id,
                                                                        supplier_table,
                                                                        part_table,
                                                                        partsupp_task,
                                                                        supplier_hash_map,
                                                                        part_hash_map));
    task_id++;
  }

  return pipeline_results;
}

std::shared_ptr<gqe::physical::relation> fused_filter_join(
  std::shared_ptr<gqe::physical::relation> supplier_table,
  std::shared_ptr<gqe::physical::relation> part_table,
  std::shared_ptr<gqe::physical::relation> partsupp_table)
{
  return std::make_shared<gqe::physical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(
      {std::move(supplier_table), std::move(part_table), std::move(partsupp_table)}),
    fused_filter_join_generate_tasks,
    /*last_child_break_pipeline=*/false);
}

}  // namespace q16
}  // namespace benchmark
}  // namespace gqe_python
