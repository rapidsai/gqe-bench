/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <pybind11/pytypes.h>
#include <tpch/q10/fused_probes_task.hpp>
#include <tpch/q10/sort_limit_task.hpp>
#include <tpch/q10/unique_key_inner_join_task.hpp>
#include <tpch/q13/filter_orders_task.hpp>
#include <tpch/q13/groupjoin_task.hpp>
#include <tpch/q16/task.hpp>
#include <tpch/q18/groupby_task.hpp>
#include <tpch/q21/task.hpp>
#include <tpch/q22/task.hpp>

#include <gqe/catalog.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/device_properties.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/memory_resource/boost_shared_memory_resource.hpp>
#include <gqe/memory_resource/memory_utilities.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/physical/set.hpp>
#include <gqe/physical/shuffle.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/cupti.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/tpch.hpp>

#include <git_revision.h.in>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

#include <cudf/datetime.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>

#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cassert>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace lib {

std::shared_ptr<gqe::physical::relation> read(std::string table_name,
                                              std::vector<std::string> column_names,
                                              gqe::expression const* partial_filter,
                                              std::vector<gqe::column_traits> column_defs)
{
  std::unordered_map<std::string, cudf::data_type> columns;
  for (auto const& [column_name, type, column_properties] : column_defs) {
    columns.insert({column_name, type});
  }
  std::vector<cudf::data_type> data_types;
  if (!columns.empty()) {
    // if column_defs is given, we need to check whether column names are in it
    for (const auto& name : column_names) {
      if (columns.count(name)) {
        data_types.push_back(columns[name]);
      } else {
        throw std::logic_error("unable to find column name " + name + " in the " + table_name +
                               " table definition");
      }
    }
  }
  return std::make_shared<gqe::physical::read_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    std::move(column_names),
    std::move(table_name),
    partial_filter ? partial_filter->clone() : nullptr,
    std::move(data_types));
}

// The following relation factories create a copy for the expression because Python cannot give up
// ownership of an object to C++, but the relation constructors accept a unique_ptr as argument.
std::shared_ptr<gqe::physical::relation> filter(std::shared_ptr<gqe::physical::relation> input,
                                                gqe::expression const* condition,
                                                std::vector<cudf::size_type> projection_indices)
{
  return std::make_shared<gqe::physical::filter_relation>(
    std::move(input),
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    condition->clone(),
    projection_indices);
}

std::shared_ptr<gqe::physical::relation> broadcast_join(
  std::shared_ptr<gqe::physical::relation> left_table,
  std::shared_ptr<gqe::physical::relation> right_table,
  gqe::expression const* condition,
  gqe::join_type_type join_type,
  std::vector<cudf::size_type> projection_indices,
  bool broadcast_left_side,
  gqe::unique_keys_policy unique_keys_pol = gqe::unique_keys_policy::none,
  bool perfect_hashing                    = false,
  gqe::expression const* left_filter      = nullptr,
  gqe::expression const* right_filter     = nullptr)
{
  gqe::physical::broadcast_policy policy = gqe::physical::broadcast_policy::right;
  if (broadcast_left_side) policy = gqe::physical::broadcast_policy::left;
  return std::make_shared<gqe::physical::broadcast_join_relation>(
    std::move(left_table),
    std::move(right_table),
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    join_type,
    condition->clone(),
    std::move(projection_indices),
    policy,
    unique_keys_pol,
    perfect_hashing,
    left_filter ? left_filter->clone() : nullptr,
    right_filter ? right_filter->clone() : nullptr);
}

std::shared_ptr<gqe::physical::relation> shuffle_join(
  std::shared_ptr<gqe::physical::relation> left_table,
  std::shared_ptr<gqe::physical::relation> right_table,
  gqe::expression const* condition,
  gqe::join_type_type join_type,
  std::vector<cudf::size_type> projection_indices,
  gqe::unique_keys_policy unique_keys_pol = gqe::unique_keys_policy::none,
  bool perfect_hashing                    = false)
{
  return std::make_shared<gqe::physical::shuffle_join_relation>(
    std::move(left_table),
    std::move(right_table),
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    join_type,
    condition->clone(),
    std::move(projection_indices),
    unique_keys_pol,
    perfect_hashing);
}

std::shared_ptr<gqe::physical::relation> shuffle(
  std::shared_ptr<gqe::physical::relation> input,
  std::vector<std::shared_ptr<gqe::expression>> shuffle_cols)
{
  std::vector<std::unique_ptr<gqe::expression>> cloned_cols;
  for (auto const& expr : shuffle_cols) {
    assert(expr->type() == gqe::expression::expression_type::column_reference);
    cloned_cols.push_back(expr->clone());
  }

  return std::make_shared<gqe::physical::shuffle_relation>(
    std::move(input),
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    std::move(cloned_cols));
}

std::shared_ptr<gqe::physical::relation> aggregate(
  std::shared_ptr<gqe::physical::relation> input,
  std::vector<std::shared_ptr<gqe::expression>> keys,
  std::vector<std::pair<cudf::aggregation::Kind, std::shared_ptr<gqe::expression>>> measures,
  gqe::expression const* condition = nullptr,
  bool perfect_hashing             = false)
{
  std::vector<std::unique_ptr<gqe::expression>> cloned_keys;
  for (auto const& key : keys) {
    cloned_keys.push_back(key->clone());
  }

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> cloned_measures;
  for (auto& [kind, expr] : measures) {
    cloned_measures.emplace_back(kind, expr->clone());
  }

  std::unique_ptr<gqe::expression> cloned_condition = condition ? condition->clone() : nullptr;

  return std::make_shared<gqe::physical::concatenate_aggregate_relation>(
    std::move(input),
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    std::move(cloned_keys),
    std::move(cloned_measures),
    std::move(cloned_condition),
    perfect_hashing);
}

std::shared_ptr<gqe::physical::relation> project(
  std::shared_ptr<gqe::physical::relation> input,
  std::vector<std::shared_ptr<gqe::expression>> output_expressions)
{
  std::vector<std::unique_ptr<gqe::expression>> cloned_expressions;
  for (auto const& expr : output_expressions) {
    cloned_expressions.push_back(expr->clone());
  }

  return std::make_shared<gqe::physical::project_relation>(
    std::move(input),
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    std::move(cloned_expressions));
}

std::shared_ptr<gqe::physical::relation> sort(
  std::shared_ptr<gqe::physical::relation> input,
  std::vector<cudf::order> column_orders,
  std::vector<cudf::null_order> null_precedences,
  std::vector<std::shared_ptr<gqe::expression>> expressions)
{
  std::vector<std::unique_ptr<gqe::expression>> cloned_expressions;
  for (auto const& expr : expressions) {
    cloned_expressions.push_back(expr->clone());
  }

  return std::make_shared<gqe::physical::concatenate_sort_relation>(
    std::move(input),
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    std::move(cloned_expressions),
    std::move(column_orders),
    std::move(null_precedences));
}

std::shared_ptr<gqe::physical::relation> fetch(std::shared_ptr<gqe::physical::relation> input,
                                               int64_t offset,
                                               int64_t count)
{
  return std::make_shared<gqe::physical::fetch_relation>(std::move(input), offset, count);
}

std::shared_ptr<gqe::physical::relation> union_all(std::shared_ptr<gqe::physical::relation> lhs,
                                                   std::shared_ptr<gqe::physical::relation> rhs)
{
  return std::make_shared<gqe::physical::union_all_relation>(std::move(lhs), std::move(rhs));
}

void mpi_init() { GQE_MPI_TRY(MPI_Init(nullptr, nullptr)); }

int mpi_rank()
{
  int rank;
  GQE_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  return rank;
}

void mpi_finalize() { GQE_MPI_TRY(MPI_Finalize()); }

void mpi_barrier() { GQE_MPI_TRY(MPI_Barrier(MPI_COMM_WORLD)); }

/*
  Multi process runtime context stores the multiprocess task manager context which is used
  to manage the nvshmem initialization, scheduler, communicator and device memory resource.
  It also stores the whether to use shared in-memory table, and initializes the shared memory if
  needed.

  We currently only initialize the multi_process_runtime_context once throughout the
  benchmarking process:
  1. Nvshmem initialization needs to happen before any CUDA calls, as they create a cuda context.
  And, pinning the shared memory calls cudaHostRegister.
  2. Allocation and pinning of shared memory take considerable time on systems like A100 (approx for
  SF1k - 5 mins) hence, we only want to do allocating and pinning the memory once.
*/

struct multi_process_runtime_context {
  multi_process_runtime_context(gqe::SCHEDULER_TYPE scheduler_type, std::string storage_kind)
  {
    _task_manager_ctx =
      gqe::multi_process_task_manager_context::default_init(MPI_COMM_WORLD, scheduler_type);

    if (storage_kind != "boost_shared_memory") { return; }

    use_in_memory_table_multigpu = true;

    // Eagerly initialize the boost shared memory resource to avoid
    // https://nvbugspro.nvidia.com/bug/5634155
    std::ignore = _task_manager_ctx->get_table_memory_resource(gqe::memory_kind::boost_shared{});
  }

  gqe::multi_process_task_manager_context* get_task_manager_ctx()
  {
    return _task_manager_ctx.get();
  }

  void finalize() { _task_manager_ctx->finalize(); }

  void update_scheduler(gqe::SCHEDULER_TYPE scheduler_type)
  {
    _task_manager_ctx->update_scheduler(scheduler_type);
  }

  std::unique_ptr<gqe::multi_process_task_manager_context> _task_manager_ctx;
  bool use_in_memory_table_multigpu = false;
};

/*
 We share the statistics after table registration (in-memory-write-tasks), this is the best way
 without requiring extra synchronization during the in-memory-write-tasks. As, other ranks
 would have to wait for the root rank to finish the in-memory-write-tasks, before reading the
 statistics.
*/
void share_statistics_interprocess(
  gqe::catalog* catalog,
  std::shared_ptr<lib::multi_process_runtime_context> multi_process_runtime_ctx)
{
  int rank;
  GQE_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  auto* mr_ptr = multi_process_runtime_ctx->get_task_manager_ctx()->get_table_memory_resource_ptr(
    gqe::memory_kind::boost_shared{});
  auto& segment =
    dynamic_cast<gqe::memory_resource::boost_shared_memory_resource*>(mr_ptr)->segment();

  if (rank == 0) {
    for (auto const& table_name : catalog->table_names()) {
      if (table_name.find("_parquet") != std::string::npos) { continue; }
      auto table_statistics_manager = catalog->statistics(table_name);
      segment.construct<gqe::table_statistics>(table_name.c_str())(
        table_statistics_manager->statistics());
      GQE_LOG_TRACE("Table on rank 0: {} has {} rows",
                    table_name,
                    table_statistics_manager->statistics().num_rows);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    for (auto const& table_name : catalog->table_names()) {
      if (table_name.find("_parquet") != std::string::npos) { continue; }
      auto table_statistics_manager = catalog->statistics(table_name);
      auto statistics               = segment.find<gqe::table_statistics>(table_name.c_str()).first;
      table_statistics_manager->add_rows(statistics->num_rows);
      GQE_LOG_TRACE("Table on rank {} has {} rows", rank, statistics->num_rows);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    for (auto const& table_name : catalog->table_names()) {
      if (table_name.find("_parquet") != std::string::npos) { continue; }
      segment.destroy<gqe::table_statistics>(table_name.c_str());
      GQE_LOG_TRACE("Table on rank {} destroyed: {}", rank, table_name);
    }
  }
}

std::shared_ptr<gqe::physical::relation> load_substrait(
  gqe::catalog* catalog,
  std::string substrait_file,
  bool optimize                                                                 = true,
  std::shared_ptr<lib::multi_process_runtime_context> multi_process_runtime_ctx = nullptr)
{
  gqe::substrait_parser parser(catalog);
  auto logical_plan_vector = parser.from_file(substrait_file);

  if (logical_plan_vector.size() > 1)
    throw std::logic_error("gqe-python only supports substrait plan with one root");

  std::shared_ptr<gqe::logical::relation> logical_plan = logical_plan_vector[0];

  if (optimize) {
    gqe::optimizer::optimization_configuration logical_rule_config(
      {gqe::optimizer::logical_optimization_rule_type::projection_pushdown,
       gqe::optimizer::logical_optimization_rule_type::string_to_int_literal,
       gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
       gqe::optimizer::logical_optimization_rule_type::join_unique_keys,
       gqe::optimizer::logical_optimization_rule_type::fix_partial_filter_column_references},
      {});
    auto optimizer =
      std::make_unique<gqe::optimizer::logical_optimizer>(&logical_rule_config, catalog);
    logical_plan = optimizer->optimize(logical_plan);
    GQE_LOG_TRACE("Optimized logical plan: \n {}", logical_plan->to_string());
  }

  if (multi_process_runtime_ctx && multi_process_runtime_ctx->use_in_memory_table_multigpu) {
    GQE_LOG_TRACE("Sharing statistics among processes");
    share_statistics_interprocess(catalog, multi_process_runtime_ctx);
  }

  gqe::physical_plan_builder plan_builder(catalog);
  auto physical_plan = plan_builder.build(logical_plan.get());
  GQE_LOG_TRACE("Generated physical plan: \n {}", physical_plan->to_string());
  return physical_plan;
}

/**
 * @brief Logs the string representation of a physical relation to a file.
 *
 * This function calls the to_string() method on the provided relation object
 * and writes the resulting string to the file specified by file_path.
 *
 * @param relation A shared_ptr to the physical relation to be logged.
 * @param file_path The path to the output file.
 * @throws std::logic_error if the file cannot be opened for writing.
 */
void log_physical_plan(std::shared_ptr<gqe::physical::relation> relation, std::string file_path)
{
  // Create an output file stream object. This attempts to open the file.
  std::ofstream output_file(file_path);

  // Check if the file stream is in a valid state (i.e., the file was opened).
  if (!output_file.is_open()) {
    // If the file could not be opened, throw an exception.
    throw std::logic_error("Error: Could not open or create file for writing at path: " +
                           file_path);
  }

  // Ensure the relation pointer is not null before dereferencing it.
  if (relation) {
    // Get the string representation and write it to the file.
    output_file << relation->to_string() << std::endl;
  }
}

struct base_context {
  virtual gqe::task_manager_context* get_task_manager_ctx()                        = 0;
  virtual py::tuple execute(gqe::catalog* catalog,
                            std::shared_ptr<gqe::physical::relation> relation,
                            std::optional<std::string> output_path = std::nullopt) = 0;
  virtual void refresh_query_context(gqe::optimization_parameters parameters)      = 0;
  virtual ~base_context() {}

 protected:
  base_context() {}
};

struct context : base_context {
  context(gqe::optimization_parameters parameters,
          bool debug_mem_usage                                  = false,
          std::optional<std::vector<std::string>> cupti_metrics = std::nullopt)
  {
    if (debug_mem_usage) {
      auto _mr =
        std::make_unique<rmm::mr::cuda_async_memory_resource>(0);  // set initial pool size to 0
      _task_manager_ctx = std::make_unique<gqe::task_manager_context>(parameters, std::move(_mr));
    } else {
      using upstream_mr = rmm::mr::cuda_memory_resource;
      using pool_mr     = rmm::mr::pool_memory_resource<upstream_mr>;
      using wrapper_mr  = rmm::mr::owning_wrapper<pool_mr, upstream_mr>;

      auto upstream = std::make_unique<upstream_mr>();
      auto mr       = std::make_unique<wrapper_mr>(
        std::move(upstream),
        parameters.initial_query_memory,
        parameters.max_query_memory.value_or(gqe::detail::default_device_memory_pool_size()));
      _task_manager_ctx = std::make_unique<gqe::task_manager_context>(parameters, std::move(mr));
    }

    _query_ctx = std::make_unique<gqe::query_context>(parameters);

    // Configure the CUPTI profiler.
    if (cupti_metrics) {
      gqe::utility::user_range_profiler::configuration profiler_config;
      profiler_config.metrics = *cupti_metrics;
      _profiler = std::make_unique<gqe::utility::user_range_profiler>(std::move(profiler_config));
    }
  }

  gqe::task_manager_context* get_task_manager_ctx() { return _task_manager_ctx.get(); }

  void refresh_query_context(gqe::optimization_parameters parameters) override
  {
    _query_ctx = std::make_unique<gqe::query_context>(parameters);
  }

  // Specifing `output_path` while `relation` does not produce an output has undefined behavior.
  // Return execution time in s.
  py::tuple execute(gqe::catalog* catalog,
                    std::shared_ptr<gqe::physical::relation> relation,
                    std::optional<std::string> output_path = std::nullopt)
  {
    gqe::task_graph_builder graph_builder(
      gqe::context_reference{_task_manager_ctx.get(), _query_ctx.get()}, catalog);
    auto task_graph = graph_builder.build(relation.get());

    // Initialize profile result outside of measurement range.
    py::dict profile;

    // Start profiling.
    if (_profiler) { _profiler->start(); }

    // Start timing.
    auto start_time = std::chrono::high_resolution_clock::now();
    execute_task_graph_single_gpu(gqe::context_reference{_task_manager_ctx.get(), _query_ctx.get()},
                                  task_graph.get());
    // Stop timing.
    auto end_time = std::chrono::high_resolution_clock::now();

    // Stop profiling.
    if (_profiler) {
      auto cupti_profile = _profiler->stop();

      // Convert profile from C++ map to Python dict.
      for (auto& metric_value : cupti_profile.metric_values) {
        profile[metric_value.first.c_str()] = metric_value.second;
      }
    }

    // Output the result to disk
    if (output_path) {
      auto destination = cudf::io::sink_info(output_path.value());
      auto options     = cudf::io::parquet_writer_options::builder(
        destination, task_graph->root_tasks[0]->result().value());
      cudf::io::write_parquet(options);
    }

    std::chrono::duration<double> elapsed_time_s = end_time - start_time;
    py::tuple performance                        = py::make_tuple(elapsed_time_s.count(), profile);

    return performance;
  }

  std::unique_ptr<gqe::query_context> _query_ctx;
  std::unique_ptr<gqe::task_manager_context> _task_manager_ctx;
  std::unique_ptr<gqe::utility::user_range_profiler> _profiler;
};

struct multi_process_context : base_context {
  multi_process_context(std::shared_ptr<lib::multi_process_runtime_context> runtime_ctx,
                        gqe::optimization_parameters parameters,
                        gqe::SCHEDULER_TYPE scheduler_type = gqe::SCHEDULER_TYPE::ROUND_ROBIN)
    : _runtime_ctx(runtime_ctx)
  {
    _task_manager_ctx = runtime_ctx->get_task_manager_ctx();
    _task_manager_ctx->update_scheduler(scheduler_type);

    parameters.use_in_memory_table_multigpu = runtime_ctx->use_in_memory_table_multigpu;

    _query_ctx = std::make_unique<gqe::query_context>(parameters);
  }

  gqe::task_manager_context* get_task_manager_ctx() { return _task_manager_ctx; }

  void refresh_query_context(gqe::optimization_parameters parameters) override
  {
    parameters.use_in_memory_table_multigpu = _runtime_ctx->use_in_memory_table_multigpu;
    _query_ctx                              = std::make_unique<gqe::query_context>(parameters);
  }

  // Specifing `output_path` while `relation` does not produce an output has undefined behavior.
  // Return execution time in ms.
  py::tuple execute(gqe::catalog* catalog,
                    std::shared_ptr<gqe::physical::relation> relation,
                    std::optional<std::string> output_path = std::nullopt)
  {
    gqe::task_graph_builder graph_builder(
      gqe::context_reference{_task_manager_ctx, _query_ctx.get()}, catalog);
    auto task_graph = graph_builder.build(relation.get());

    // Barrier sync to ensure all processes are ready to execute
    GQE_MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));

    auto start_time = std::chrono::high_resolution_clock::now();
    execute_task_graph_multi_process(gqe::context_reference{_task_manager_ctx, _query_ctx.get()},
                                     task_graph.get());
    auto end_time = std::chrono::high_resolution_clock::now();

    // Output the result to disk
    if (output_path && _task_manager_ctx->comm->rank() == 0) {
      auto destination = cudf::io::sink_info(output_path.value());
      auto options     = cudf::io::parquet_writer_options::builder(
        destination, task_graph->root_tasks[0]->result().value());
      cudf::io::write_parquet(options);
    }

    // Wait for result to be written to disk
    GQE_MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));

    std::chrono::duration<double> elapsed_time_s = end_time - start_time;
    py::tuple performance = py::make_tuple(elapsed_time_s.count(), py::none());

    return performance;
  }

  std::shared_ptr<lib::multi_process_runtime_context> _runtime_ctx;
  std::unique_ptr<gqe::query_context> _query_ctx;
  gqe::multi_process_task_manager_context* _task_manager_ctx;
};

void register_tables_parquet(
  gqe::catalog* catalog,
  std::string dataset_location,
  std::unordered_map<std::string, std::vector<gqe::column_traits>> table_definitions)
{
  for (auto const& [name, definition] : table_definitions) {
    auto const file_paths = gqe::utility::get_parquet_files(dataset_location + "/" + name);

    catalog->register_table(name,
                            definition,
                            gqe::storage_kind::parquet_file{file_paths},
                            gqe::partitioning_schema_kind::automatic{});
  }
}

// TODO: duplicate code with tpc.cpp
gqe::storage_kind::type parse_storage_kind(const std::string& storage_kind_description,
                                           const std::vector<std::string>& file_paths)
{
  std::string normalized_description;
  normalized_description.reserve(storage_kind_description.size());
  std::transform(storage_kind_description.begin(),
                 storage_kind_description.end(),
                 std::back_inserter(normalized_description),
                 [](auto c) { return std::tolower(c); });
  if (normalized_description == "system_memory") {
    return gqe::storage_kind::system_memory{};
  } else if (normalized_description == "numa_memory") {
    return gqe::storage_kind::numa_memory();  // Do not set NUMA node affinity to enable
                                              // auto-configuration.
  } else if (normalized_description == "pinned_memory") {
    return gqe::storage_kind::pinned_memory{};
  } else if (normalized_description == "numa_pinned_memory") {
    return gqe::storage_kind::numa_pinned_memory();  // Do not set NUMA node affinity to enable
                                                     // auto-configuration.
  } else if (normalized_description == "device_memory") {
    return gqe::storage_kind::device_memory{rmm::cuda_device_id(0)};
  } else if (normalized_description == "managed_memory") {
    return gqe::storage_kind::managed_memory{};
  } else if (normalized_description == "parquet_file") {
    return gqe::storage_kind::parquet_file{file_paths};
  } else if (normalized_description == "boost_shared_memory") {
    return gqe::storage_kind::boost_shared_memory{};
  }
  throw std::logic_error("Unrecognized storage kind: " + storage_kind_description);
}

void register_table_in_memory(lib::base_context* ctx,
                              gqe::catalog* catalog,
                              std::string name,
                              std::vector<gqe::column_traits> const& definition,
                              std::vector<std::string> const& file_paths,
                              gqe::storage_kind::type storage_kind)
{
  catalog->register_table(name + "_parquet",
                          definition,
                          gqe::storage_kind::parquet_file{file_paths},
                          gqe::partitioning_schema_kind::automatic{});

  catalog->register_table(name, definition, storage_kind, gqe::partitioning_schema_kind::none{});

  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;
  for (auto const& [column_name, type, column_properties] : definition) {
    column_names.push_back(column_name);
    column_types.push_back(type);
  }

  auto read_table = std::make_shared<gqe::physical::read_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    column_names,
    name + "_parquet",
    nullptr,
    column_types);

  auto write_table =
    std::make_shared<gqe::physical::write_relation>(read_table, column_names, name);

  ctx->execute(catalog, write_table, std::nullopt);
}

void register_tables_in_memory(
  lib::base_context* ctx,
  gqe::catalog* catalog,
  std::string dataset_location,
  std::unordered_map<std::string, std::vector<gqe::column_traits>> table_definitions,
  const std::string& storage_kind_description)
{
  gqe::storage_kind::type storage_kind = parse_storage_kind(storage_kind_description, {});
  GQE_LOG_TRACE("Storage kind created: {}", storage_kind_description);

  for (auto const& [name, definition] : table_definitions) {
    auto const file_paths = gqe::utility::get_parquet_files(dataset_location + "/" + name);
    register_table_in_memory(ctx, catalog, name, definition, file_paths, storage_kind);
  }
}

// Get the compression statistics for all tables registered in the catalog. This includes the names
// of the compressed and uncompressed columns for each table.
std::unordered_map<std::string, gqe::table_statistics> get_table_stats(gqe::catalog* catalog)
{
  auto table_names = catalog->table_names();
  std::unordered_map<std::string, gqe::table_statistics> stats;
  for (const auto& table_name : table_names) {
    auto table_statistics_manager = catalog->statistics(table_name);
    auto table_statistics         = table_statistics_manager->statistics();
    stats[table_name]             = table_statistics;
  }
  return stats;
}

// Construct a date object from an integer representing days since the unix epoch
gqe::literal_expression<cudf::timestamp_D> date_from_days(cudf::timestamp_D::rep days)
{
  return gqe::literal_expression<cudf::timestamp_D>(cudf::timestamp_D(cudf::duration_D(days)));
}

void finalize_shared_memory()
{
  boost::interprocess::shared_memory_object::remove("gqe_shared_memory");
}

void initialize_shared_memory(size_t pool_size = 10ULL * 1024 * 1024 * 1024)
{
  // Shared memory is backed by file, and it persists even after all the processes have finished.
  // If the previous run fails to clean up the shared memory, it needs to be cleaned up here.
  try {
    if (mpi_rank() == 0) { finalize_shared_memory(); }
  } catch (const std::exception& e) {
    GQE_LOG_TRACE("Tried cleaning up shared memory: {}", e.what());
  }

  mpi_barrier();

  if (mpi_rank() == 0) {
    auto segment = boost::interprocess::managed_shared_memory(
      boost::interprocess::create_only, "gqe_shared_memory", pool_size);
  }

  mpi_barrier();
}

}  // namespace lib

PYBIND11_MODULE(lib, py_module)
{
  py_module.doc() = "Python binding for GQE library";

  // Version
  py_module.attr("libgqe_branch") = py::str(std::string(GQE_GIT_BRANCH));
  py_module.attr("libgqe_commit") = py::str(std::string(GQE_GIT_SHA1));
#ifdef GQE_GIT_IS_DIRTY
  py_module.attr("libgqe_is_dirty") = 1;
#else
  py_module.attr("libgqe_is_dirty") = 0;
#endif

  // Types

  py::enum_<gqe::join_type_type>(py_module, "JoinType")
    .value("inner", gqe::join_type_type::inner)
    .value("left", gqe::join_type_type::left)
    .value("left_semi", gqe::join_type_type::left_semi)
    .value("left_anti", gqe::join_type_type::left_anti)
    .value("full", gqe::join_type_type::full);

  py::enum_<cudf::aggregation::Kind>(py_module, "AggregationKind")
    .value("sum", cudf::aggregation::SUM)
    .value("avg", cudf::aggregation::MEAN)
    .value("count_all", cudf::aggregation::COUNT_ALL)
    .value("count_valid", cudf::aggregation::COUNT_VALID)
    .value("min", cudf::aggregation::MIN)
    .value("max", cudf::aggregation::MAX);

  py::enum_<cudf::order>(py_module, "Order")
    .value("ascending", cudf::order::ASCENDING)
    .value("descending", cudf::order::DESCENDING);

  py::enum_<cudf::null_order>(py_module, "NullOrder")
    .value("after", cudf::null_order::AFTER)
    .value("before", cudf::null_order::BEFORE);

  py::enum_<cudf::type_id>(py_module, "TypeId")
    .value("int8", cudf::type_id::INT8)
    .value("int32", cudf::type_id::INT32)
    .value("int64", cudf::type_id::INT64)
    .value("float32", cudf::type_id::FLOAT32)
    .value("float64", cudf::type_id::FLOAT64)
    .value("string", cudf::type_id::STRING)
    .value("timestamp_days", cudf::type_id::TIMESTAMP_DAYS);

  py::enum_<gqe::column_traits::column_property>(py_module, "ColumnProperty")
    .value("unique", gqe::column_traits::column_property::unique);

  py::enum_<gqe::unique_keys_policy>(py_module, "UniqueKeysPolicy")
    .value("none", gqe::unique_keys_policy::none)
    .value("right", gqe::unique_keys_policy::right)
    .value("left", gqe::unique_keys_policy::left);

  py::class_<cudf::data_type>(py_module, "DataType")
    .def(py::init<cudf::type_id>())
    .def(py::init<cudf::type_id, int32_t>())
    .def("type_id", &cudf::data_type::id)
    .def("scale", &cudf::data_type::scale);

  py::enum_<cudf::datetime::datetime_component>(py_module, "DateTimeComponent")
    .value("year", cudf::datetime::datetime_component::YEAR)
    .value("month", cudf::datetime::datetime_component::MONTH)
    .value("day", cudf::datetime::datetime_component::DAY)
    .value("weekday", cudf::datetime::datetime_component::WEEKDAY)
    .value("hour", cudf::datetime::datetime_component::HOUR)
    .value("minute", cudf::datetime::datetime_component::MINUTE)
    .value("second", cudf::datetime::datetime_component::SECOND)
    .value("millisecond", cudf::datetime::datetime_component::MILLISECOND)
    .value("nanosecond", cudf::datetime::datetime_component::NANOSECOND);

  // Optimization Parameters
  py::enum_<gqe::compression_format>(py_module, "CompressionFormat")
    .value("none", gqe::compression_format::none)
    .value("ans", gqe::compression_format::ans)
    .value("lz4", gqe::compression_format::lz4)
    .value("snappy", gqe::compression_format::snappy)
    .value("gdeflate", gqe::compression_format::gdeflate)
    .value("deflate", gqe::compression_format::deflate)
    .value("cascaded", gqe::compression_format::cascaded)
    .value("zstd", gqe::compression_format::zstd)
    .value("gzip", gqe::compression_format::gzip)
    .value("bitcomp", gqe::compression_format::bitcomp)
    .value("best_compression_ratio", gqe::compression_format::best_compression_ratio)
    .value("best_decompression_speed", gqe::compression_format::best_decompression_speed);

  py::enum_<gqe::io_engine_type>(py_module, "IoEngineType")
    .value("automatic", gqe::io_engine_type::automatic)
    .value("io_uring", gqe::io_engine_type::io_uring)
    .value("psync", gqe::io_engine_type::psync);

  py::enum_<nvcompType_t>(py_module, "NvcompType")
    .value("char", NVCOMP_TYPE_CHAR)
    .value("uchar", NVCOMP_TYPE_UCHAR)
    .value("short", NVCOMP_TYPE_SHORT)
    .value("ushort", NVCOMP_TYPE_USHORT)
    .value("int", NVCOMP_TYPE_INT)
    .value("uint", NVCOMP_TYPE_UINT)
    .value("longlong", NVCOMP_TYPE_LONGLONG)
    .value("ulonglong", NVCOMP_TYPE_ULONGLONG)
    .value("bits", NVCOMP_TYPE_BITS);

  py::class_<gqe::optimization_parameters>(py_module, "OptimizationParameters")
    .def(py::init<bool>(), py::arg("only_defaults") = false)
    .def_readwrite("max_num_workers", &gqe::optimization_parameters::max_num_workers)
    .def_readwrite("max_num_partitions", &gqe::optimization_parameters::max_num_partitions)
    .def_readwrite("log_level", &gqe::optimization_parameters::log_level)
    .def_readwrite("initial_query_memory", &gqe::optimization_parameters::initial_query_memory)
    .def_readwrite("max_query_memory", &gqe::optimization_parameters::max_query_memory)
    .def_readwrite("initial_task_manager_memory",
                   &gqe::optimization_parameters::initial_task_manager_memory)
    .def_readwrite("max_task_manager_memory",
                   &gqe::optimization_parameters::max_task_manager_memory)
    .def_readwrite("join_use_hash_map_cache",
                   &gqe::optimization_parameters::join_use_hash_map_cache)
    .def_readwrite("join_use_unique_keys", &gqe::optimization_parameters::join_use_unique_keys)
    .def_readwrite("join_use_perfect_hash", &gqe::optimization_parameters::join_use_perfect_hash)
    .def_readwrite("join_use_mark_join", &gqe::optimization_parameters::join_use_mark_join)
    .def_readwrite("read_zero_copy_enable", &gqe::optimization_parameters::read_zero_copy_enable)
    .def_readwrite("use_customized_io", &gqe::optimization_parameters::use_customized_io)
    .def_readwrite("io_bounce_buffer_size", &gqe::optimization_parameters::io_bounce_buffer_size)
    .def_readwrite("io_auxiliary_threads", &gqe::optimization_parameters::io_auxiliary_threads)
    .def_readwrite("use_opt_type_for_single_char_col",
                   &gqe::optimization_parameters::use_opt_type_for_single_char_col)
    .def_readwrite("use_in_memory_table_multigpu",
                   &gqe::optimization_parameters::use_in_memory_table_multigpu)
    .def_readwrite("in_memory_table_compression_format",
                   &gqe::optimization_parameters::in_memory_table_compression_format)
    .def_readwrite("in_memory_table_compression_chunk_size",
                   &gqe::optimization_parameters::in_memory_table_compression_chunk_size)
    .def_readwrite("in_memory_table_compression_ratio_threshold",
                   &gqe::optimization_parameters::in_memory_table_compression_ratio_threshold)
    .def_readwrite("in_memory_table_secondary_compression_format",
                   &gqe::optimization_parameters::in_memory_table_secondary_compression_format)
    .def_readwrite(
      "in_memory_table_secondary_compression_ratio_threshold",
      &gqe::optimization_parameters::in_memory_table_secondary_compression_ratio_threshold)
    .def_readwrite(
      "in_memory_table_secondary_compression_multiplier_threshold",
      &gqe::optimization_parameters::in_memory_table_secondary_compression_multiplier_threshold)
    .def_readwrite("in_memory_table_compression_ratio_threshold",
                   &gqe::optimization_parameters::in_memory_table_compression_ratio_threshold)
    .def_readwrite("in_memory_table_use_cpu_compression",
                   &gqe::optimization_parameters::use_cpu_compression)
    .def_readwrite("in_memory_table_compression_level",
                   &gqe::optimization_parameters::compression_level)
    .def_readwrite("io_block_size", &gqe::optimization_parameters::io_block_size)
    .def_readwrite("io_engine", &gqe::optimization_parameters::io_engine)
    .def_readwrite("io_pipelining", &gqe::optimization_parameters::io_pipelining)
    .def_readwrite("io_alignment", &gqe::optimization_parameters::io_alignment)
    .def_readwrite("use_overlap_mtx", &gqe::optimization_parameters::use_overlap_mtx)
    .def_readwrite("use_partition_pruning", &gqe::optimization_parameters::use_partition_pruning)
    .def_readwrite("zone_map_partition_size",
                   &gqe::optimization_parameters::zone_map_partition_size)
    .def_readwrite("filter_use_like_shift_and",
                   &gqe::optimization_parameters::filter_use_like_shift_and)
    .def_readwrite("aggregation_use_perfect_hash",
                   &gqe::optimization_parameters::aggregation_use_perfect_hash)
    .def_readwrite("num_shuffle_partitions", &gqe::optimization_parameters::num_shuffle_partitions);

  // Catalog
  py::class_<gqe::catalog>(py_module, "Catalog")
    .def(py::init([](lib::base_context* ctx) -> gqe::catalog {
      return gqe::catalog(ctx->get_task_manager_ctx());
    }))
    .def("column_names", &gqe::catalog::column_names)
    .def("table_names", &gqe::catalog::table_names)
    .def("statistics", &gqe::catalog::statistics);
  // Table statistics manager
  py::class_<gqe::table_statistics_manager>(py_module, "TableStatisticsManager")
    .def("statistics", &gqe::table_statistics_manager::statistics);
  py::class_<gqe::column_traits>(py_module, "ColumnTraits")
    .def(py::init<std::string const&,
                  cudf::data_type const&,
                  std::vector<gqe::column_traits::column_property> const&>())
    .def(py::init<std::string const&, cudf::data_type const&>())
    .def_readwrite("name", &gqe::column_traits::name)
    .def_readwrite("data_type", &gqe::column_traits::data_type)
    .def_readwrite("is_unique", &gqe::column_traits::is_unique);
  py_module.def("register_tables_parquet", &lib::register_tables_parquet);
  py_module.def("register_tables_in_memory", &lib::register_tables_in_memory);
  py_module.def("get_table_stats", &lib::get_table_stats);

  // Table statistics
  py::class_<gqe::table_statistics>(py_module, "TableStatistics")
    .def_readonly("num_rows", &gqe::table_statistics::num_rows)
    .def_readonly("num_columns", &gqe::table_statistics::num_columns)
    .def_readonly("num_row_groups", &gqe::table_statistics::num_row_groups)
    .def_readonly("compressed_num_row_groups", &gqe::table_statistics::compressed_num_row_groups)
    .def_readonly("compressed_size_per_column", &gqe::table_statistics::compressed_size_per_column)
    .def_readonly("uncompressed_size_per_column",
                  &gqe::table_statistics::uncompressed_size_per_column);

  // Relations
  py::class_<gqe::physical::relation, std::shared_ptr<gqe::physical::relation>> relation_cls(
    py_module, "Relation");

  py_module.def("read", &lib::read);
  py_module.def("filter", &lib::filter);
  py_module.def("broadcast_join", &lib::broadcast_join);
  py_module.def("shuffle_join", &lib::shuffle_join);
  py_module.def("aggregate", &lib::aggregate);
  py_module.def("project", &lib::project);
  py_module.def("sort", &lib::sort);
  py_module.def("fetch", &lib::fetch);
  py_module.def("union_all", &lib::union_all);
  py_module.def("shuffle", &lib::shuffle);
  py_module.def("load_substrait", &lib::load_substrait);
  py_module.def("log_physical_plan", &lib::log_physical_plan);
  py_module.def("q10_fused_probes_join_map_build",
                &gqe_python::benchmark::q10::fused_probes_join_map_build);
  py_module.def("q10_fused_probes_join_multimap_build",
                &gqe_python::benchmark::q10::fused_probes_join_multimap_build);
  py_module.def("q10_fused_probes_join_probe",
                &gqe_python::benchmark::q10::fused_probes_join_probe);
  py_module.def("q10_sort_limit", &gqe_python::benchmark::q10::sort_limit);
  py_module.def("q10_unique_key_inner_join_build",
                &gqe_python::benchmark::q10::unique_key_inner_join_build);
  py_module.def("q10_unique_key_inner_join_probe",
                &gqe_python::benchmark::q10::unique_key_inner_join_probe);
  py_module.def("q13_groupjoin_build", &gqe_python::benchmark::q13::groupjoin_build);
  py_module.def("q13_groupjoin_probe", &gqe_python::benchmark::q13::groupjoin_probe);
  py_module.def("q13_groupjoin_retrieve", &gqe_python::benchmark::q13::groupjoin_retrieve);
  py_module.def("q13_filter_orders", &gqe_python::benchmark::q13::filter_orders);
  py_module.def("q13_fused_filter_probe", &gqe_python::benchmark::q13::fused_filter_probe);
  py_module.def("q16_fused_filter_join", &gqe_python::benchmark::q16::fused_filter_join);
  py_module.def("q16_aggregate", &gqe_python::benchmark::q16::aggregate);
  py_module.def("q18_groupby", &gqe_python::benchmark::q18::groupby);
  py_module.def("q21_left_anti_join", &gqe_python::benchmark::q21::left_anti_join_probe);
  py_module.def("q21_left_semi_join", &gqe_python::benchmark::q21::left_semi_join_probe);
  py_module.def("q21_left_anti_join_retrieve",
                &gqe_python::benchmark::q21::left_anti_join_retrieve);
  py_module.def("q21_left_semi_join_retrieve",
                &gqe_python::benchmark::q21::left_semi_join_retrieve);
  py_module.def("q22_fused_project_filter", &gqe_python::benchmark::q22::fused_project_filter);
  py_module.def("q22_mark_join", &gqe_python::benchmark::q22::mark_join);

  // Expressions
  py::class_<gqe::expression, std::shared_ptr<gqe::expression>> expr_cls(py_module, "Expression");
  py::class_<gqe::column_reference_expression, std::shared_ptr<gqe::column_reference_expression>>(
    py_module, "ColumnReference", expr_cls)
    .def(py::init<cudf::size_type>());
  py::class_<gqe::cast_expression, std::shared_ptr<gqe::cast_expression>>(
    py_module, "Cast", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, cudf::data_type>());

  // Binary expressions
  py::class_<gqe::equal_expression, std::shared_ptr<gqe::equal_expression>>(
    py_module, "Equal", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::not_equal_expression, std::shared_ptr<gqe::not_equal_expression>>(
    py_module, "NotEqual", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::less_expression, std::shared_ptr<gqe::less_expression>>(
    py_module, "Less", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::greater_expression, std::shared_ptr<gqe::greater_expression>>(
    py_module, "Greater", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::less_equal_expression, std::shared_ptr<gqe::less_equal_expression>>(
    py_module, "LessEqual", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::greater_equal_expression, std::shared_ptr<gqe::greater_equal_expression>>(
    py_module, "GreaterEqual", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());

  py::class_<gqe::logical_and_expression, std::shared_ptr<gqe::logical_and_expression>>(
    py_module, "And", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());

  py::class_<gqe::logical_or_expression, std::shared_ptr<gqe::logical_or_expression>>(
    py_module, "Or", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());

  py::class_<gqe::multiply_expression, std::shared_ptr<gqe::multiply_expression>>(
    py_module, "Multiply", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::divide_expression, std::shared_ptr<gqe::divide_expression>>(
    py_module, "Divide", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::add_expression, std::shared_ptr<gqe::add_expression>>(py_module, "Add", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::subtract_expression, std::shared_ptr<gqe::subtract_expression>>(
    py_module, "Subtract", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());

  // Scalar functions
  py::class_<gqe::like_expression, std::shared_ptr<gqe::like_expression>>(
    py_module, "Like", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::string, std::string, bool>());

  py::class_<gqe::substr_expression, std::shared_ptr<gqe::substr_expression>>(
    py_module, "Substr", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, cudf::size_type, cudf::size_type>());

  // If-then-else expression
  py::class_<gqe::if_then_else_expression, std::shared_ptr<gqe::if_then_else_expression>>(
    py_module, "IfThenElse", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>,
                  std::shared_ptr<gqe::expression>,
                  std::shared_ptr<gqe::expression>>());

  // Date-part expression
  py::class_<gqe::datepart_expression, std::shared_ptr<gqe::datepart_expression>>(
    py_module, "DatePart", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, cudf::datetime::datetime_component>());

  // Literals
  py::class_<gqe::literal_expression<std::string>,
             std::shared_ptr<gqe::literal_expression<std::string>>>(
    py_module, "LiteralString", expr_cls)
    .def(py::init<std::string, bool>());
  py::class_<gqe::literal_expression<float>, std::shared_ptr<gqe::literal_expression<float>>>(
    py_module, "LiteralFloat", expr_cls)
    .def(py::init<float, bool>());
  py::class_<gqe::literal_expression<double>, std::shared_ptr<gqe::literal_expression<double>>>(
    py_module, "LiteralDouble", expr_cls)
    .def(py::init<double, bool>());
  py::class_<gqe::literal_expression<int32_t>, std::shared_ptr<gqe::literal_expression<int32_t>>>(
    py_module, "LiteralInt32", expr_cls)
    .def(py::init<int32_t, bool>());
  py::class_<gqe::literal_expression<int64_t>, std::shared_ptr<gqe::literal_expression<int64_t>>>(
    py_module, "LiteralInt64", expr_cls)
    .def(py::init<int64_t, bool>());
  py::class_<gqe::literal_expression<cudf::timestamp_D>,
             std::shared_ptr<gqe::literal_expression<cudf::timestamp_D>>>(
    py_module, "LiteralTimestampD", expr_cls);

  py_module.def("date_from_days", &lib::date_from_days);

  // Base context class - needed for Catalog to accept both Context and MultiProcessContext
  py::class_<lib::base_context, std::shared_ptr<lib::base_context>>(py_module, "BaseContext");

  py::class_<lib::context, lib::base_context, std::shared_ptr<lib::context>>(py_module, "Context")
    .def(py::init<gqe::optimization_parameters,            // parameters
                  bool,                                    // debug_mem_usage
                  std::optional<std::vector<std::string>>  // cupti_metrics
                  >(),
         py::arg("parameters"),
         py::arg("debug_mem_usage") = false,
         py::arg("cupti_metrics")   = std::nullopt)
    .def("execute", &lib::context::execute)
    .def("refresh_query_context", &lib::context::refresh_query_context);

  py_module.def("mpi_init", &lib::mpi_init);
  py_module.def("mpi_finalize", &lib::mpi_finalize);
  py_module.def("mpi_rank", &lib::mpi_rank);
  py_module.def("mpi_barrier", &lib::mpi_barrier);

  py_module.def("initialize_shared_memory", &lib::initialize_shared_memory);
  py_module.def("finalize_shared_memory", &lib::finalize_shared_memory);

  py::enum_<gqe::SCHEDULER_TYPE>(py_module, "scheduler_type")
    .value("ALL_TO_ALL", gqe::SCHEDULER_TYPE::ALL_TO_ALL)
    .value("ROUND_ROBIN", gqe::SCHEDULER_TYPE::ROUND_ROBIN);

  py::class_<lib::multi_process_runtime_context,
             std::shared_ptr<lib::multi_process_runtime_context>>(py_module,
                                                                  "MultiProcessRuntimeContext")
    .def(py::init<gqe::SCHEDULER_TYPE, std::string>())
    .def("get_task_manager_ctx", &lib::multi_process_runtime_context::get_task_manager_ctx)
    .def("finalize", &lib::multi_process_runtime_context::finalize)
    .def("update_scheduler", &lib::multi_process_runtime_context::update_scheduler);

  py::class_<lib::multi_process_context,
             lib::base_context,
             std::shared_ptr<lib::multi_process_context>>(py_module, "MultiProcessContext")
    .def(py::init<std::shared_ptr<lib::multi_process_runtime_context>,  // runtime_ctx
                  gqe::optimization_parameters,                         // parameters
                  gqe::SCHEDULER_TYPE                                   // scheduler_type
                  >(),
         py::arg("runtime_ctx"),
         py::arg("parameters"),
         py::arg("scheduler_type") = gqe::SCHEDULER_TYPE::ROUND_ROBIN)
    .def("execute", &lib::multi_process_context::execute)
    .def("refresh_query_context", &lib::multi_process_context::refresh_query_context);
}
