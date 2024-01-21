/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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
#include <gqe/executor/query_context.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/tpch.hpp>

#include <cudf/io/parquet.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace lib {

std::shared_ptr<gqe::logical::relation> read(gqe::catalog* catalog,
                                             std::string table_name,
                                             std::vector<std::string> column_names)
{
  std::vector<cudf::data_type> column_types;
  column_types.reserve(column_names.size());
  for (auto const& column_name : column_names) {
    column_types.push_back(catalog->column_type(table_name, column_name));
  }

  return std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(column_names),
    std::move(column_types),
    std::move(table_name),
    nullptr);
}

// The following relation factories create a copy for the expression because Python cannot give up
// ownership of an object to C++, but the relation constructors accept a unique_ptr as argument.
std::shared_ptr<gqe::logical::relation> filter(std::shared_ptr<gqe::logical::relation> input,
                                               gqe::expression const* condition)
{
  return std::make_shared<gqe::logical::filter_relation>(
    std::move(input), std::vector<std::shared_ptr<gqe::logical::relation>>(), condition->clone());
}

std::shared_ptr<gqe::logical::relation> join(std::shared_ptr<gqe::logical::relation> left,
                                             std::shared_ptr<gqe::logical::relation> right,
                                             std::shared_ptr<gqe::expression> condition,
                                             gqe::join_type_type join_type,
                                             std::vector<cudf::size_type> projection_indices)
{
  return std::make_shared<gqe::logical::join_relation>(
    std::move(left),
    std::move(right),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    condition->clone(),
    join_type,
    std::move(projection_indices));
}

std::shared_ptr<gqe::logical::relation> aggregate(
  std::shared_ptr<gqe::logical::relation> input,
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

  return std::make_shared<gqe::logical::aggregate_relation>(
    std::move(input),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(cloned_keys),
    std::move(cloned_measures));
}

std::shared_ptr<gqe::logical::relation> project(
  std::shared_ptr<gqe::logical::relation> input,
  std::vector<std::shared_ptr<gqe::expression>> output_expressions)
{
  std::vector<std::unique_ptr<gqe::expression>> cloned_expressions;
  for (auto const& expr : output_expressions) {
    cloned_expressions.push_back(expr->clone());
  }

  return std::make_shared<gqe::logical::project_relation>(
    std::move(input),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(cloned_expressions));
}

// Specifing `output_result=true` while `relation` does not produce an output has undefined
// behavior.
void execute(gqe::catalog* catalog,
             std::shared_ptr<gqe::logical::relation> relation,
             bool output_result = true,
             bool log_time      = true)
{
  gqe::physical_plan_builder plan_builder(catalog);
  auto physical_plan = plan_builder.build(relation.get());

  gqe::query_context qctx(gqe::optimization_parameters{});

  gqe::task_graph_builder graph_builder(&qctx, catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  if (log_time) {
    gqe::utility::time_function(gqe::execute_task_graph_single_gpu, &qctx, task_graph.get());
  } else {
    execute_task_graph_single_gpu(&qctx, task_graph.get());
  }

  // Output the result to disk
  if (output_result) {
    auto destination = cudf::io::sink_info("output.parquet");
    auto options     = cudf::io::parquet_writer_options::builder(
      destination, task_graph->root_tasks[0]->result().value());
    cudf::io::write_parquet(options);
  }
}

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

void register_tpch_in_memory(gqe::catalog* catalog, std::string dataset_location)
{
  auto const& table_definitions = gqe::utility::tpch::table_definitions();
  for (auto const& [name, definition] : table_definitions) {
    auto const file_paths = gqe::utility::get_parquet_files(dataset_location + "/" + name);

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

    auto read_table = std::make_shared<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>(),
      column_names,
      column_types,
      name + "_parquet",
      nullptr);

    auto write_table =
      std::make_shared<gqe::logical::write_relation>(read_table, column_names, column_types, name);

    execute(catalog, write_table, false, false);
  }
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
    .value("avg", cudf::aggregation::MEAN);

  // Catalog
  py::class_<gqe::catalog>(py_module, "Catalog").def(py::init<>());
  py_module.def("register_tpch_parquet", &lib::register_tpch_parquet);
  py_module.def("register_tpch_in_memory", &lib::register_tpch_in_memory);

  // Relations
  py::class_<gqe::logical::relation, std::shared_ptr<gqe::logical::relation>> relation_cls(
    py_module, "Relation");

  py_module.def("read", &lib::read);
  py_module.def("filter", &lib::filter);
  py_module.def("join", &lib::join);
  py_module.def("aggregate", &lib::aggregate);
  py_module.def("project", &lib::project);

  // Expressions
  py::class_<gqe::expression, std::shared_ptr<gqe::expression>> expr_cls(py_module, "Expression");
  py::class_<gqe::column_reference_expression, std::shared_ptr<gqe::column_reference_expression>>(
    py_module, "ColumnReference", expr_cls)
    .def(py::init<cudf::size_type>());

  // Binary expressions
  py::class_<gqe::equal_expression, std::shared_ptr<gqe::equal_expression>>(
    py_module, "Equal", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::logical_and_expression, std::shared_ptr<gqe::logical_and_expression>>(
    py_module, "And", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::less_expression, std::shared_ptr<gqe::less_expression>>(
    py_module, "Less", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::multiply_expression, std::shared_ptr<gqe::multiply_expression>>(
    py_module, "Multiply", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());
  py::class_<gqe::divide_expression, std::shared_ptr<gqe::divide_expression>>(
    py_module, "Divide", expr_cls)
    .def(py::init<std::shared_ptr<gqe::expression>, std::shared_ptr<gqe::expression>>());

  // Literals
  py::class_<gqe::literal_expression<std::string>,
             std::shared_ptr<gqe::literal_expression<std::string>>>(
    py_module, "LiteralString", expr_cls)
    .def(py::init<std::string, bool>());
  py::class_<gqe::literal_expression<double>, std::shared_ptr<gqe::literal_expression<double>>>(
    py_module, "LiteralDouble", expr_cls)
    .def(py::init<double, bool>());

  // Execution
  py_module.def("execute", &lib::execute);
}
