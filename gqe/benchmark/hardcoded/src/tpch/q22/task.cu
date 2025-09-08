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

#include <tpch/q22/fused_project_filter.cuh>
#include <tpch/q22/mark_join.cuh>
#include <tpch/q22/task.hpp>
#include <utility/utility.hpp>

namespace gqe_python {
namespace benchmark {
namespace q22 {

fused_project_filter_task::fused_project_filter_task(gqe::context_reference ctx_ref,
                                                     int32_t task_id,
                                                     int32_t stage_id,
                                                     std::shared_ptr<gqe::task> customer_table)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(customer_table)}, {})
{
}

/**
 * @brief Functor to launch the fused kernel for projection and filtering.
 */
struct fused_kernel_functor {
  cudf::column_device_view c_phone_column;
  cudf::column_device_view c_acctbal_column;
  cudf::column_device_view c_custkey_column;
  cudf::size_type* d_counter_ptr;
  char* out_c_phone_column;
  cudf::mutable_column_device_view out_c_acctbal_column;
  cudf::mutable_column_device_view out_c_custkey_column;
  rmm::cuda_stream_view main_stream;

  template <typename T1, typename T2>
  void operator()() const
  {
    // Only allow int32/int64 here if you want
    if constexpr ((std::is_same_v<T1, int32_t> || std::is_same_v<T1, int64_t>) &&
                  (std::is_same_v<T2, float> || std::is_same_v<T2, double>)) {
      auto grid_size = gqe_python::utility::find_grid_size(fused_project_filter_kernel<T1, T2>,
                                                           gqe_python::utility::block_dim);
      fused_project_filter_kernel<T1, T2>
        <<<grid_size, gqe_python::utility::block_dim, 0, main_stream>>>(c_phone_column,
                                                                        c_acctbal_column,
                                                                        c_custkey_column,
                                                                        d_counter_ptr,
                                                                        out_c_phone_column,
                                                                        out_c_acctbal_column,
                                                                        out_c_custkey_column);
    } else {
      CUDF_FAIL("First column must be INT32 or INT64 and second column must be FLOAT OR DOUBLE");
    }
  }
};

// A customized task to use a fused kernel for the projection and filter operations.
// c_acctbal > 0.00 and substring(c_phone from 1 for 2) in ('13', '31', '23', '29', '30', '18',
// '17')
void fused_project_filter_task::execute()
{
  auto const main_stream = cudf::get_default_stream();
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto customer_table  = dependent_tasks[0]->result().value();
  auto const row_count = customer_table.num_rows();

  auto c_custkey_column = cudf::column_device_view::create(customer_table.column(0), main_stream);
  auto c_phone_column   = cudf::column_device_view::create(customer_table.column(1), main_stream);
  auto c_acctbal_column = cudf::column_device_view::create(customer_table.column(2), main_stream);

  cudf::data_type identifier_type = c_custkey_column->type();
  cudf::data_type decimal_type    = c_acctbal_column->type();

  // Conservative to allocate the size.
  constexpr int32_t num_char_per_phone = 2;
  auto out_c_phone_column = rmm::device_uvector<char>(num_char_per_phone * row_count, main_stream);

  auto out_c_acctbal_column =
    cudf::make_numeric_column(decimal_type, row_count, cudf::mask_state::UNALLOCATED, main_stream);
  auto out_c_acctbal_view =
    cudf::mutable_column_device_view::create(out_c_acctbal_column->mutable_view(), main_stream);
  auto out_c_custkey_column = cudf::make_numeric_column(
    identifier_type, row_count, cudf::mask_state::UNALLOCATED, main_stream);
  auto out_c_custkey_view =
    cudf::mutable_column_device_view::create(out_c_custkey_column->mutable_view(), main_stream);

  rmm::device_scalar<cudf::size_type> d_counter(0, main_stream);
  cudf::size_type* d_counter_ptr = d_counter.data();

  // Launch the fused kernel for projection and filtering.
  cudf::double_type_dispatcher(identifier_type,
                               decimal_type,
                               fused_kernel_functor{*c_phone_column,
                                                    *c_acctbal_column,
                                                    *c_custkey_column,
                                                    d_counter_ptr,
                                                    out_c_phone_column.data(),
                                                    *out_c_acctbal_view,
                                                    *out_c_custkey_view,
                                                    main_stream});
  GQE_CUDA_TRY(cudaGetLastError());

  main_stream.synchronize();
  cudf::size_type h_counter = d_counter.value(main_stream);

  std::vector<std::unique_ptr<cudf::column>> out_columns;

  auto string_col = make_customized_string_column(std::move(out_c_phone_column), h_counter);
  auto acctbal_col =
    std::make_unique<cudf::column>(cudf::slice(*out_c_acctbal_column, {0, h_counter})[0]);
  auto custkey_col =
    std::make_unique<cudf::column>(cudf::slice(*out_c_custkey_column, {0, h_counter})[0]);
  out_columns.push_back(std::move(custkey_col));
  out_columns.push_back(std::move(string_col));
  out_columns.push_back(std::move(acctbal_col));

  auto new_results = std::make_unique<cudf::table>(std::move(out_columns));
  emit_result(std::move(new_results));

  remove_dependencies();
}

// Functor for generating the output tasks from input tasks.
std::vector<std::shared_ptr<gqe::task>> fused_project_filter_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id)
{
  std::shared_ptr<gqe::task> customer_table;
  if (children_tasks[0].size() == 1) {
    customer_table = children_tasks[0][0];
  } else {
    customer_table =
      std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[0]);
    task_id++;
  }
  std::vector<std::shared_ptr<gqe::task>> pipeline_results;
  pipeline_results.push_back(
    std::make_shared<fused_project_filter_task>(ctx_ref, task_id, stage_id, customer_table));

  return pipeline_results;
}

mark_join_task::mark_join_task(gqe::context_reference ctx_ref,
                               int32_t task_id,
                               int32_t stage_id,
                               std::shared_ptr<gqe::task> customer_table,
                               std::shared_ptr<gqe::task> orders_table)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(customer_table), std::move(orders_table)}, {})
{
}

/**
 * @brief Functor to launch the mark join kernel.
 *
 */
struct mark_join_functor {
  cudf::column_device_view c_custkey_column;
  cudf::column_device_view o_custkey_column;
  rmm::cuda_stream_view main_stream;
  cudf::size_type customer_table_size;
  cudf::size_type* d_scan_counter_ptr;
  cudf::size_type* customer_out_indices;

  template <typename identifier_type>
  void operator()() const
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<identifier_type, int32_t> ||
                  std::is_same_v<identifier_type, int64_t>) {
      auto constexpr load_factor = 0.5;
      mark_join_map_type<identifier_type> mark_join_map{
        customer_table_size,
        load_factor,
        empty_key_sentinel<identifier_type>,
        empty_value_sentinel,
        {},
        {},
        {},
        {},
        gqe_python::utility::map_allocator_type<identifier_type>{
          gqe_python::utility::map_allocator_instance_type<identifier_type>{}, main_stream}};

      // Create bloom filter.
      // Hyperparameters for bloom filter size to achieve best performance.
      constexpr int32_t bf_size_factor = 4;
      gqe_python::utility::bloom_filter_type<identifier_type> bloom_filter(
        gqe_python::utility::get_bloom_filter_blocks<
          gqe_python::utility::bloom_filter_type<identifier_type>>(customer_table_size *
                                                                   bf_size_factor),
        {},
        {},
        gqe_python::utility::bloom_filter_allocator_type{
          gqe_python::utility::bloom_filter_allocator_instance_type{}, main_stream});

      // Insert c_custkey into left anti join map.
      thrust::for_each(thrust::make_counting_iterator<cudf::size_type>(0),
                       thrust::make_counting_iterator<cudf::size_type>(c_custkey_column.size()),
                       [map  = mark_join_map.ref(cuco::insert),
                        keys = c_custkey_column] __device__(auto row_idx) mutable {
                         // We don't need to check for NULLs here.
                         identifier_type key =
                           static_cast<identifier_type>(keys.element<identifier_type>(row_idx));
                         map.insert(thrust::pair<identifier_type, cudf::size_type>(key, row_idx));
                       });

      auto map_device_view = mark_join_map.ref(cuco::find, cuco::for_each);

      // Add to bloom filter with type conversion.
      auto it = thrust::make_transform_iterator(
        thrust::make_counting_iterator<cudf::size_type>(0),
        [keys = c_custkey_column] __device__(auto row_idx) -> identifier_type {
          return static_cast<identifier_type>(keys.element<identifier_type>(row_idx));
        });
      bloom_filter.add(it, it + c_custkey_column.size(), main_stream);
      auto bloom_filter_ref = bloom_filter.ref();

      // Global counters.
      rmm::device_scalar<cudf::size_type> d_probe_counter(0, main_stream);
      cudf::size_type* d_probe_counter_ptr = d_probe_counter.data();

      // Probe kernel with explicit template parameter.
      auto probe_grid_size = gqe_python::utility::find_grid_size(mark_join_kernel<identifier_type>,
                                                                 gqe_python::utility::block_dim);
      mark_join_kernel<identifier_type>
        <<<probe_grid_size, gqe_python::utility::block_dim, 0, main_stream>>>(
          map_device_view, bloom_filter_ref, o_custkey_column, d_probe_counter_ptr);

      // Scan kernel.
      auto scan_grid_size = gqe_python::utility::find_grid_size(iterate_join_map<identifier_type>,
                                                                gqe_python::utility::block_dim);
      iterate_join_map<identifier_type>
        <<<scan_grid_size, gqe_python::utility::block_dim, 0, main_stream>>>(
          map_device_view, customer_out_indices, d_scan_counter_ptr);

    } else {
      CUDF_FAIL("Column must be INT32 or INT64");
    }
  }
};

void mark_join_task::execute()
{
  auto const main_stream = cudf::get_default_stream();
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  // Get the input tables
  auto customer_table = dependencies()[0]->result().value();
  auto orders_table   = dependencies()[1]->result().value();

  // Determine the hash table size.
  auto const customer_table_size = customer_table.num_rows();

  auto c_custkey_column = cudf::column_device_view::create(customer_table.column(0), main_stream);
  auto o_custkey_column = cudf::column_device_view::create(orders_table.column(0), main_stream);
  cudf::data_type identifier_type = c_custkey_column->type();

  // The result size is capped by the size of customer table.
  rmm::device_uvector<cudf::size_type> customer_out_indices(customer_table_size, main_stream);

  rmm::device_scalar<cudf::size_type> d_scan_counter(0, main_stream);
  cudf::size_type* d_scan_counter_ptr = d_scan_counter.data();

  // Use type dispatcher to handle runtime type
  cudf::type_dispatcher(identifier_type,
                        mark_join_functor{*c_custkey_column,
                                          *o_custkey_column,
                                          main_stream,
                                          customer_table_size,
                                          d_scan_counter_ptr,
                                          customer_out_indices.data()});

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
                               static_cast<void const*>(customer_out_indices.data()),
                               nullptr,
                               0};
  std::vector<std::unique_ptr<cudf::column>> out_columns;
  out_columns.push_back(materialize_column(customer_table, 1, gather_map));
  out_columns.push_back(materialize_column(customer_table, 2, gather_map));
  emit_result(std::make_unique<cudf::table>(std::move(out_columns)));
  remove_dependencies();
}

// Functor for generating the output tasks from input tasks.
std::vector<std::shared_ptr<gqe::task>> mark_join_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id)
{
  std::shared_ptr<gqe::task> customer_table;
  // Don't bother concatenating if the number of row groups is only one.
  if (children_tasks[0].size() == 1) {
    customer_table = children_tasks[0][0];
  } else {
    customer_table =
      std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[0]);
    task_id++;
  }

  std::shared_ptr<gqe::task> orders_table;
  if (children_tasks[1].size() == 1) {
    orders_table = children_tasks[1][0];
  } else {
    orders_table =
      std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[1]);
    task_id++;
  }
  std::vector<std::shared_ptr<gqe::task>> pipeline_results;
  pipeline_results.push_back(
    std::make_shared<mark_join_task>(ctx_ref, task_id, stage_id, customer_table, orders_table));

  return pipeline_results;
}

std::shared_ptr<gqe::physical::relation> fused_project_filter(
  std::shared_ptr<gqe::physical::relation> input)
{
  return std::make_shared<gqe::physical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>({std::move(input)}),
    fused_project_filter_generate_tasks,
    false);
}

std::shared_ptr<gqe::physical::relation> mark_join(
  std::shared_ptr<gqe::physical::relation> customer_table,
  std::shared_ptr<gqe::physical::relation> orders_table)
{
  return std::make_shared<gqe::physical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(
      {std::move(customer_table), std::move(orders_table)}),
    mark_join_generate_tasks,
    false);
}

}  // namespace q22
}  // namespace benchmark
}  // namespace gqe_python
