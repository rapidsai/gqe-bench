/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/physical/set.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/tpch.hpp>

#include <gqe/optimizer/logical_optimization.hpp>

#include <cudf/datetime.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <stdexcept>

namespace py = pybind11;

namespace lib {

std::shared_ptr<gqe::physical::relation> read(std::string table_name,
                                              std::vector<std::string> column_names,
                                              gqe::expression const* partial_filter)
{
  return std::make_shared<gqe::physical::read_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>(),
    std::move(column_names),
    std::move(table_name),
    partial_filter ? partial_filter->clone() : nullptr);
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
  gqe::unique_keys_policy unique_keys_pol = gqe::unique_keys_policy::none)
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
    unique_keys_pol);
}

std::shared_ptr<gqe::physical::relation> aggregate(
  std::shared_ptr<gqe::physical::relation> input,
  std::vector<std::shared_ptr<gqe::expression>> keys,
  std::vector<std::pair<cudf::aggregation::Kind, std::shared_ptr<gqe::expression>>> measures,
  gqe::expression const* condition = nullptr)
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
    std::move(cloned_condition));
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
  gqe::physical_plan_builder plan_builder(catalog);
  return plan_builder.build(logical_plan.get());
}

struct context {
  context(int32_t max_num_workers                           = 1,
          int32_t max_num_partitions                        = 8,
          std::string in_memory_table_compression_format    = "none",
          std::string in_memory_table_compression_data_type = "char",
          int32_t compression_chunk_size                    = 65536,
          bool use_opt_type_for_single_char_col             = true,
          bool use_overlap_mtx                              = true,
          bool join_use_hash_map_cache                      = false,
          bool read_use_zero_copy                           = false,
          bool join_use_unique_keys                         = true,
          bool debug_mem_usage                              = false)
  {
    if (debug_mem_usage) {
      auto _mr = std::make_unique<rmm::mr::cuda_async_memory_resource>(0); // set initial pool size to 0
      _task_manager_ctx = std::make_unique<gqe::task_manager_context>(std::move(_mr));
    } else {
      using upstream_mr_type = rmm::mr::cuda_memory_resource;
      using mr_type          = rmm::mr::pool_memory_resource<upstream_mr_type>;
      auto pool_size = gqe::utility::default_device_memory_pool_size();
      auto upstream_mr = std::make_shared<upstream_mr_type>();
      auto mr = std::make_unique<rmm::mr::owning_wrapper<mr_type, upstream_mr_type>>(upstream_mr, pool_size, pool_size);
      _task_manager_ctx = std::make_unique<gqe::task_manager_context>(std::move(mr));
    }

    gqe::optimization_parameters parameters(false);
    parameters.max_num_workers                  = max_num_workers;
    parameters.max_num_partitions               = max_num_partitions;
    parameters.use_opt_type_for_single_char_col = use_opt_type_for_single_char_col;
    parameters.use_overlap_mtx                  = use_overlap_mtx;
    parameters.join_use_hash_map_cache          = join_use_hash_map_cache;
    parameters.read_zero_copy_enable            = read_use_zero_copy;
    parameters.join_use_unique_keys             = join_use_unique_keys;

    // FIXME: DRY compression format
    if (in_memory_table_compression_format == "none") {
      parameters.in_memory_table_compression_format = gqe::compression_format::none;
    } else if (in_memory_table_compression_format == "ans") {
      parameters.in_memory_table_compression_format = gqe::compression_format::ans;
    } else if (in_memory_table_compression_format == "lz4") {
      parameters.in_memory_table_compression_format = gqe::compression_format::lz4;
    } else if (in_memory_table_compression_format == "snappy") {
      parameters.in_memory_table_compression_format = gqe::compression_format::snappy;
    } else if (in_memory_table_compression_format == "gdeflate") {
      parameters.in_memory_table_compression_format = gqe::compression_format::gdeflate;
    } else if (in_memory_table_compression_format == "deflate") {
      parameters.in_memory_table_compression_format = gqe::compression_format::deflate;
    } else if (in_memory_table_compression_format == "cascaded") {
      parameters.in_memory_table_compression_format = gqe::compression_format::cascaded;
    } else if (in_memory_table_compression_format == "zstd") {
      parameters.in_memory_table_compression_format = gqe::compression_format::zstd;
    } else if (in_memory_table_compression_format == "gzip") {
      parameters.in_memory_table_compression_format = gqe::compression_format::gzip;
    } else if (in_memory_table_compression_format == "bitcomp") {
      parameters.in_memory_table_compression_format = gqe::compression_format::bitcomp;
    } else if (in_memory_table_compression_format == "best_compression_ratio") {
      parameters.in_memory_table_compression_format =
        gqe::compression_format::best_compression_ratio;
    } else if (in_memory_table_compression_format == "best_decompression_speed") {
      parameters.in_memory_table_compression_format =
        gqe::compression_format::best_decompression_speed;
    } else {
      throw std::logic_error("Unrecognized compression format");
    }

    // FIXME: DRY compression data type
    if (in_memory_table_compression_data_type == "char") {
      parameters.in_memory_table_compression_data_type = NVCOMP_TYPE_CHAR;
    } else if (in_memory_table_compression_data_type == "short") {
      parameters.in_memory_table_compression_data_type = NVCOMP_TYPE_SHORT;
    } else if (in_memory_table_compression_data_type == "int") {
      parameters.in_memory_table_compression_data_type = NVCOMP_TYPE_INT;
    } else if (in_memory_table_compression_data_type == "longlong") {
      parameters.in_memory_table_compression_data_type = NVCOMP_TYPE_LONGLONG;
    } else if (in_memory_table_compression_data_type == "bits") {
      parameters.in_memory_table_compression_data_type = NVCOMP_TYPE_BITS;
    } else {
      throw std::logic_error("Unrecognized data type format");
    }

    parameters.compression_chunk_size = compression_chunk_size;

    _query_ctx = std::make_unique<gqe::query_context>(parameters);
  }

  // Specifing `output_path` while `relation` does not produce an output has undefined behavior.
  // Return execution time in ms.
  float execute(gqe::catalog* catalog,
                std::shared_ptr<gqe::physical::relation> relation,
                std::optional<std::string> output_path = std::nullopt)
  {
    gqe::task_graph_builder graph_builder(
      gqe::context_reference{_task_manager_ctx.get(), _query_ctx.get()}, catalog);
    auto task_graph = graph_builder.build(relation.get());

    auto start_time = std::chrono::high_resolution_clock::now();
    execute_task_graph_single_gpu(gqe::context_reference{_task_manager_ctx.get(), _query_ctx.get()},
                                  task_graph.get());
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

  std::unique_ptr<gqe::query_context> _query_ctx;
  std::unique_ptr<gqe::task_manager_context> _task_manager_ctx;
};

void register_tpch_parquet(
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
  std::map<std::string, gqe::storage_kind::type> const storage_kinds{
    {"system_memory", gqe::storage_kind::system_memory{}},
    {"numa_memory", gqe::storage_kind::numa_memory{gqe::cpu_set(0)}},
    {"pinned_memory", gqe::storage_kind::pinned_memory{}},
    {"numa_pinned_memory", gqe::storage_kind::numa_pinned_memory{gqe::cpu_set(0)}},
    {"device_memory", gqe::storage_kind::device_memory{rmm::cuda_device_id(0)}},
    {"managed_memory", gqe::storage_kind::managed_memory{}},
    {"parquet_file", gqe::storage_kind::parquet_file{file_paths}}};
  return storage_kinds.at(normalized_description);
}

void register_table_in_memory(gqe::catalog* catalog,
                              int32_t num_row_groups,
                              std::string in_memory_table_compression_format,
                              std::string in_memory_table_compression_data_type,
                              int32_t compression_chunk_size,
                              std::string name,
                              std::vector<gqe::column_traits> const& definition,
                              std::vector<std::string> const& file_paths,
                              const std::string& storage_kind_description)
{
  catalog->register_table(name + "_parquet",
                          definition,
                          gqe::storage_kind::parquet_file{file_paths},
                          gqe::partitioning_schema_kind::automatic{});

  catalog->register_table(name,
                          definition,
                          parse_storage_kind(storage_kind_description, file_paths),
                          gqe::partitioning_schema_kind::none{});

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
    nullptr);

  auto write_table =
    std::make_shared<gqe::physical::write_relation>(read_table, column_names, name);

  context ctx(1,
              num_row_groups,
              in_memory_table_compression_format,
              in_memory_table_compression_data_type,
              compression_chunk_size);

  ctx.execute(catalog, write_table, std::nullopt);
}

void register_tpch_in_memory(
  gqe::catalog* catalog,
  std::string dataset_location,
  int32_t num_row_groups,
  std::string in_memory_table_compression_format,
  std::string in_memory_table_compression_data_type,
  int32_t compression_chunk_size,
  std::unordered_map<std::string, std::vector<gqe::column_traits>> table_definitions,
  const std::string& storage_kind_description)
{
  for (auto const& [name, definition] : table_definitions) {
    auto const file_paths = gqe::utility::get_parquet_files(dataset_location + "/" + name);
    register_table_in_memory(catalog,
                             num_row_groups,
                             in_memory_table_compression_format,
                             in_memory_table_compression_data_type,
                             compression_chunk_size,
                             name,
                             definition,
                             file_paths,
                             storage_kind_description);
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

  // Catalog
  py::class_<gqe::catalog>(py_module, "Catalog").def(py::init<>());
  py::class_<gqe::column_traits>(py_module, "ColumnTraits")
    .def(py::init<std::string const&,
                  cudf::data_type const&,
                  std::vector<gqe::column_traits::column_property> const&>())
    .def(py::init<std::string const&, cudf::data_type const&>())
    .def_readwrite("name", &gqe::column_traits::name)
    .def_readwrite("data_type", &gqe::column_traits::data_type)
    .def_readwrite("is_unique", &gqe::column_traits::is_unique);
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
  py_module.def("union_all", &lib::union_all);
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

  // Execution
  py::class_<lib::context, std::shared_ptr<lib::context>>(py_module, "Context")
    .def(py::init<int32_t,
                  int32_t,
                  std::string,
                  std::string,
                  int32_t,
                  bool,
                  bool,
                  bool,
                  bool,
                  bool,
                  bool>())
    .def("execute", &lib::context::execute);
}
