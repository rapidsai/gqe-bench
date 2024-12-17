/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/catalog.hpp>
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
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/query_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/tpch.hpp>

#include <gqe/optimizer/logical_optimization.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "table_definitions.hpp"
#include <chrono>
#include <stdexcept>

namespace py = pybind11;

namespace lib {

std::shared_ptr<gqe::physical::relation> read(std::string table_name,
                                              std::vector<std::string> column_names)
{
  return std::make_shared<gqe::physical::read_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    std::move(column_names),
    std::move(table_name),
    nullptr);
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
  bool broadcast_left_side)
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
    policy);
}

std::shared_ptr<gqe::physical::relation> aggregate(
  std::shared_ptr<gqe::physical::relation> input,
  std::vector<std::shared_ptr<gqe::expression>> keys,
  std::vector<std::pair<cudf::aggregation::Kind, std::shared_ptr<gqe::expression>>> measures)
{
  std::vector<std::unique_ptr<gqe::expression>> cloned_keys;
  for (auto const& key : keys) {
    cloned_keys.push_back(key->clone());
  }

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> cloned_measures;
  for (auto& [kind, expr] : measures) {
    cloned_measures.emplace_back(kind, expr->clone());
  }

  return std::make_shared<gqe::physical::concatenate_aggregate_relation>(
    std::move(input),
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    std::move(cloned_keys),
    std::move(cloned_measures));
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

std::shared_ptr<gqe::physical::relation> load_substrait(gqe::catalog* catalog,
                                                        std::string substrait_file,
                                                        bool optimize = true)
{
  gqe::substrait_parser parser(catalog);
  auto logical_plan_vector = parser.from_file(substrait_file);

  if (logical_plan_vector.size() > 1)
    throw std::logic_error("gqe-python only supports substrait plan with one root");

  std::shared_ptr<gqe::logical::relation> logical_plan = logical_plan_vector[0];

  if (optimize) {
    gqe::optimizer::optimization_configuration logical_rule_config(
      {gqe::optimizer::logical_optimization_rule_type::projection_pushdown}, {});
    auto optimizer =
      std::make_unique<gqe::optimizer::logical_optimizer>(&logical_rule_config, catalog);
    logical_plan = optimizer->optimize(logical_plan);
    GQE_LOG_TRACE("Optimized logical plan: \n {}", logical_plan->to_string());
  }
  gqe::physical_plan_builder plan_builder(catalog);
  return plan_builder.build(logical_plan.get());
}

struct context {
  context(int32_t max_num_workers    = 1,
          int32_t max_num_partitions = 8,
          bool read_zero_copy_enable = false)
  {
    // RMM requires the memory location to be aligned to 256B. So here, we set the memory pool size
    // to ~90% to the total memory and a multiple of 256.
    std::size_t free_memory, total_memory;
    GQE_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
    auto const pool_size = total_memory / 284 * 256;

    _pool_mr = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
      &_cuda_mr, pool_size, pool_size);
    rmm::mr::set_current_device_resource(_pool_mr.get());

    gqe::optimization_parameters parameters;
    parameters.max_num_workers       = max_num_workers;
    parameters.max_num_partitions    = max_num_partitions;
    parameters.read_zero_copy_enable = read_zero_copy_enable;

    _qctx = std::make_unique<gqe::query_context>(parameters);
  }

  // Specifing `output_path` while `relation` does not produce an output has undefined behavior.
  // Return execution time in ms.
  float execute(gqe::catalog* catalog,
                std::shared_ptr<gqe::physical::relation> relation,
                std::optional<std::string> output_path = std::nullopt)
  {
    gqe::task_graph_builder graph_builder(_qctx.get(), catalog);
    auto task_graph = graph_builder.build(relation.get());

    auto start_time = std::chrono::high_resolution_clock::now();
    execute_task_graph_single_gpu(_qctx.get(), task_graph.get());
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed_time_ms = end_time - start_time;

    // Output the result to disk
    if (output_path) {
      auto destination = cudf::io::sink_info(output_path.value());
      auto options     = cudf::io::parquet_writer_options::builder(
        destination, task_graph->root_tasks[0]->result().value());
      cudf::io::write_parquet(options);
    }

    return elapsed_time_ms.count();
  }

  std::unique_ptr<gqe::query_context> _qctx;
  rmm::mr::cuda_memory_resource _cuda_mr;
  std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> _pool_mr;
};

void register_tpch_parquet(gqe::catalog* catalog, std::string dataset_location)
{
  auto const& table_definitions = gqe::utility::tpch::table_definitions();
  for (auto const& [name, definition] : table_definitions) {
    auto const file_paths = gqe::utility::get_parquet_files(dataset_location + "/" + name);

    catalog->register_table(name,
                            definition,
                            gqe::storage_kind::parquet_file{file_paths},
                            gqe::partitioning_schema_kind::automatic{});
  }
}

void register_table_in_memory(
  gqe::catalog* catalog,
  int32_t num_row_groups,
  std::string name,
  std::vector<std::pair<std::string, cudf::data_type>> const& definition,
  std::vector<std::string> const& file_paths)
{
  catalog->register_table(name + "_parquet",
                          definition,
                          gqe::storage_kind::parquet_file{file_paths},
                          gqe::partitioning_schema_kind::automatic{});

  catalog->register_table(
    name, definition, gqe::storage_kind::pinned_memory{}, gqe::partitioning_schema_kind::none{});

  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;
  for (auto const& [column_name, type] : definition) {
    column_names.push_back(column_name);
    column_types.push_back(type);
  }

  auto read_table = std::make_shared<gqe::physical::read_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    column_names,
    name + "_parquet",
    nullptr);

  auto write_table =
    std::make_shared<gqe::physical::write_relation>(read_table, column_names, name);

  context ctx(1, num_row_groups);
  ctx.execute(catalog, write_table, std::nullopt);
}

void register_tpch_in_memory(gqe::catalog* catalog,
                             std::string dataset_location,
                             int32_t num_row_groups,
                             int load_data_of_query = 0)
{
  std::unordered_map<std::string, std::vector<tpch::column_definition_type>> table_definitions;
  if (!load_data_of_query) {
    // Load all the tables
    table_definitions = gqe::utility::tpch::table_definitions();
  } else {
    // Load only the required tables and columns
    table_definitions = query_table_definitions(load_data_of_query);
  }
  for (auto const& [name, definition] : table_definitions) {
    auto const file_paths = gqe::utility::get_parquet_files(dataset_location + "/" + name);
    register_table_in_memory(catalog, num_row_groups, name, definition, file_paths);
  }
}

// Construct a date object from an integer representing days since the unix epoch
gqe::literal_expression<cudf::timestamp_D> date_from_days(cudf::timestamp_D::rep days)
{
  return gqe::literal_expression<cudf::timestamp_D>(cudf::timestamp_D(cudf::duration_D(days)));
}

}  // namespace lib

PYBIND11_MODULE(lib, py_module)
{
  py_module.doc() = "Python binding for GQE library";

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
    .value("int64", cudf::type_id::INT64)
    .value("float64", cudf::type_id::FLOAT64);

  py::class_<cudf::data_type>(py_module, "DataType")
    .def(py::init<cudf::type_id>())
    .def(py::init<cudf::type_id, int32_t>());

  // Catalog
  py::class_<gqe::catalog>(py_module, "Catalog").def(py::init<>());
  py_module.def("register_tpch_parquet", &lib::register_tpch_parquet);
  py_module.def("register_tpch_in_memory", &lib::register_tpch_in_memory);

  // Relations
  py::class_<gqe::physical::relation, std::shared_ptr<gqe::physical::relation>> relation_cls(
    py_module, "Relation");

  py_module.def("read", &lib::read);
  py_module.def("filter", &lib::filter);
  py_module.def("broadcast_join", &lib::broadcast_join);
  py_module.def("aggregate", &lib::aggregate);
  py_module.def("project", &lib::project);
  py_module.def("sort", &lib::sort);
  py_module.def("fetch", &lib::fetch);
  py_module.def("load_substrait", &lib::load_substrait);

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

  // If-then-else expression
  py::class_<gqe::if_then_else_expression, std::shared_ptr<gqe::if_then_else_expression>>(
    py_module, "IfThenElse", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>,
                  std::shared_ptr<gqe::expression>,
                  std::shared_ptr<gqe::expression>>());

  // Literals
  py::class_<gqe::literal_expression<std::string>,
             std::shared_ptr<gqe::literal_expression<std::string>>>(
    py_module, "LiteralString", expr_cls)
    .def(py::init<std::string, bool>());
  py::class_<gqe::literal_expression<double>, std::shared_ptr<gqe::literal_expression<double>>>(
    py_module, "LiteralDouble", expr_cls)
    .def(py::init<double, bool>());
  py::class_<gqe::literal_expression<int64_t>, std::shared_ptr<gqe::literal_expression<int64_t>>>(
    py_module, "LiteralInt64", expr_cls)
    .def(py::init<int64_t, bool>());
  py::class_<gqe::literal_expression<cudf::timestamp_D>,
             std::shared_ptr<gqe::literal_expression<cudf::timestamp_D>>>(
    py_module, "LiteralTimestampD", expr_cls);

  py_module.def("date_from_days", &lib::date_from_days);

  // Execution
  py::class_<lib::context, std::shared_ptr<lib::context>>(py_module, "Context")
    .def(py::init<int32_t, int32_t, bool>())
    .def("execute", &lib::context::execute);
}
