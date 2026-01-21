/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <tpch/q18/groupby.cuh>
#include <tpch/q18/groupby_task.hpp>
#include <utility/config.hpp>
#include <utility/hash_map_cache.hpp>
#include <utility/utility.hpp>
#include <utility/write_buffer.cuh>

#include <gqe/context_reference.hpp>
#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/utility/cuda.hpp>

#include <cuda/atomic>
#include <cuda/std/optional>
#include <cuda/std/tuple>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

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
#include <memory>
#include <vector>

namespace gqe_python {
namespace benchmark {
namespace q18 {

namespace {

/**
 * @brief TPC-H Q18 filter threshold for sum(l_quantity).
 */
constexpr double q18_quantity_threshold = 300.0;

// Use the shared task_hash_map utility
using task_hash_map = gqe_python::utility::task_hash_map;

// Q18-specific hash map type with bucket_size = 1
template <typename Identifier>
using groupby_map_type = gqe_python::utility::q18::map_type<Identifier, double>;

/**
 * @brief A task for the group by aggregation.
 */
class groupby_task : public gqe::task {
 public:
  /**
   * @brief Construct a group by task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input_task The lineitem table read task.
   * @param[in] hash_map The shared hash map for aggregation across tasks.
   * @param[in] scale_factor The TPC-H scale factor for cardinality estimation.
   */
  groupby_task(gqe::context_reference ctx_ref,
               int32_t task_id,
               int32_t stage_id,
               std::shared_ptr<gqe::task>&& input_task,
               std::shared_ptr<task_hash_map> hash_map,
               double scale_factor);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::shared_ptr<task_hash_map> _hash_map;
  double _scale_factor;
};

/**
 * @brief A task for the group by retrieve.
 */
class groupby_retrieve_task : public gqe::task {
 public:
  /**
   * @brief Construct a group by retrieve task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] groupby_task The concatenated group by tasks.
   * @param[in] hash_map The hash map shared among tasks.
   * @param[in] scale_factor The TPC-H scale factor for sizing estimates.
   */
  groupby_retrieve_task(gqe::context_reference ctx_ref,
                        int32_t task_id,
                        int32_t stage_id,
                        std::shared_ptr<gqe::task> groupby_task,
                        std::shared_ptr<task_hash_map> hash_map,
                        double scale_factor);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::shared_ptr<task_hash_map> _hash_map;
  double _scale_factor;
};

groupby_task::groupby_task(gqe::context_reference ctx_ref,
                           int32_t task_id,
                           int32_t stage_id,
                           std::shared_ptr<gqe::task>&& input_task,
                           std::shared_ptr<task_hash_map> hash_map,
                           double scale_factor)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(input_task)}, {}),
    _hash_map(std::move(hash_map)),
    _scale_factor(scale_factor)
{
}

template <typename Identifier, typename MapRef>
void groupby_aggregate(cudf::column_device_view l_orderkey_column,
                       cudf::column_device_view l_quantity_column,
                       MapRef map_ref,
                       rmm::cuda_stream_view stream)
{
  const auto num_rows = l_orderkey_column.size();

  // Get raw pointers to avoid repeated metadata lookups and bounds checks
  auto order_ptr    = l_orderkey_column.data<Identifier>();
  auto quantity_ptr = l_quantity_column.data<double>();

  thrust::for_each(
    thrust::cuda::par.on(stream.value()),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(num_rows),
    [=] __device__(cudf::size_type row_idx) mutable {
      Identifier key = order_ptr[row_idx];
      double qty     = quantity_ptr[row_idx];

      // Single-pass atomic aggregation: insert (key, qty) or atomically add qty to existing value
      map_ref.insert_or_apply(
        cuco::pair<Identifier, double>(key, qty),
        [] __device__(cuda::atomic_ref<double, cuda::thread_scope_device> existing_value,
                      double new_value) {
          existing_value.fetch_add(new_value, cuda::std::memory_order_relaxed);
        });
    });
}

struct groupby_functor {
  template <typename Identifier>
  void operator()(cudf::column_device_view l_orderkey_column,
                  cudf::column_device_view l_quantity_column,
                  task_hash_map& hash_map_wrapper,
                  rmm::cuda_stream_view stream,
                  size_t cardinality,
                  double load_factor)
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      // Lazy thread-safe instantiation of hash map using utility API
      auto& hash_map = hash_map_wrapper.get_map<Identifier, double, groupby_map_type<Identifier>>(
        cardinality, load_factor);
      auto map_ref = hash_map.ref(cuco::insert_or_apply);

      groupby_aggregate<Identifier>(l_orderkey_column, l_quantity_column, map_ref, stream);
      GQE_CUDA_TRY(cudaGetLastError());
    } else {
      CUDF_FAIL("Unsupported identifier type for groupby aggregation. Expected INT32 or INT64.");
    }
  }
};

void groupby_task::execute()
{
  auto const stream = cudf::get_default_stream();
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto lineitem_table = dependent_tasks[0]->result().value();

  auto l_orderkey_column = cudf::column_device_view::create(lineitem_table.column(0), stream);
  auto l_quantity_column = cudf::column_device_view::create(lineitem_table.column(1), stream);

  cudf::data_type identifier_type = l_orderkey_column->type();

  // Cardinality estimate based on scale factor:
  // Expect ~1.5M unique orderkeys per scale factor
  size_t cardinality = 1'500'000ul * _scale_factor;
  double load_factor = 0.5;

  cudf::type_dispatcher(identifier_type,
                        groupby_functor{},
                        *l_orderkey_column,
                        *l_quantity_column,
                        *_hash_map,
                        stream,
                        cardinality,
                        load_factor);

  // Emit empty table since aggregation results are stored in shared hash map
  auto empty_table = std::make_unique<cudf::table>();
  emit_result(std::move(empty_table));

  remove_dependencies();
}

struct groupby_retrieve_functor {
  template <typename Identifier>
  void operator()(gqe::context_reference ctx_ref,
                  cudf::size_type* d_global_offset,
                  cudf::mutable_column_device_view out_l_orderkey,
                  cudf::mutable_column_device_view out_sum_quantity,
                  task_hash_map& hash_map_wrapper,
                  rmm::cuda_stream_view stream)
  {
    // Only allow int32/int64 here
    if constexpr (std::is_same_v<Identifier, int32_t> || std::is_same_v<Identifier, int64_t>) {
      auto& hash_map = hash_map_wrapper.get_map<Identifier, double, groupby_map_type<Identifier>>();
      auto hash_map_ref = hash_map.ref(cuco::find, cuco::for_each);

      auto grid_size = gqe::utility::detect_launch_grid_size(groupby_retrieve_kernel<Identifier>,
                                                             gqe_python::utility::block_dim,
                                                             /* dynamic_shared_memory_bytes = */ 0);
      groupby_retrieve_kernel<Identifier><<<grid_size, gqe_python::utility::block_dim, 0, stream>>>(
        hash_map_ref, d_global_offset, out_l_orderkey, out_sum_quantity);
      GQE_CUDA_TRY(cudaGetLastError());
    } else {
      CUDF_FAIL("Unsupported identifier type for groupby retrieval. Expected INT32 or INT64.");
    }
  }
};

groupby_retrieve_task::groupby_retrieve_task(gqe::context_reference ctx_ref,
                                             int32_t task_id,
                                             int32_t stage_id,
                                             std::shared_ptr<gqe::task> groupby_task,
                                             std::shared_ptr<task_hash_map> hash_map,
                                             double scale_factor)
  : gqe::task(ctx_ref, task_id, stage_id, {std::move(groupby_task)}, {}),
    _hash_map(std::move(hash_map)),
    _scale_factor(scale_factor)
{
}

void groupby_retrieve_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto const stream = cudf::get_default_stream();

  // Estimate output size - for sf100, the output row count is ~4k rows
  auto const estimated_row_count_retrieve = 5'000ul * _scale_factor;
  auto identifier_type                    = _hash_map->identifier_type();
  auto quantity_type                      = cudf::data_type(cudf::type_id::FLOAT64);

  auto out_l_orderkey_column = cudf::make_numeric_column(
    identifier_type, estimated_row_count_retrieve, cudf::mask_state::UNALLOCATED, stream);
  auto out_sum_quantity_column = cudf::make_numeric_column(
    quantity_type, estimated_row_count_retrieve, cudf::mask_state::UNALLOCATED, stream);

  auto out_l_orderkey_view =
    cudf::mutable_column_device_view::create(out_l_orderkey_column->mutable_view(), stream);
  auto out_sum_quantity_view =
    cudf::mutable_column_device_view::create(out_sum_quantity_column->mutable_view(), stream);

  rmm::device_scalar<cudf::size_type> d_global_offset(0, stream);

  cudf::type_dispatcher(identifier_type,
                        groupby_retrieve_functor{},
                        get_context_reference(),
                        d_global_offset.data(),
                        *out_l_orderkey_view,
                        *out_sum_quantity_view,
                        *_hash_map,
                        stream);

  cudf::size_type h_result_rows = d_global_offset.value(stream);

  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.reserve(2);

  result_columns.push_back(std::make_unique<cudf::column>(
    cudf::slice(out_l_orderkey_column->view(), {0, h_result_rows})[0]));
  result_columns.push_back(std::make_unique<cudf::column>(
    cudf::slice(out_sum_quantity_column->view(), {0, h_result_rows})[0]));

  auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
  emit_result(std::move(result_table));

  remove_dependencies();
  // Reset the hash map to free memory immediately after the task is completed
  _hash_map.reset();
}

/**
 * @brief Functor for generating the group by tasks.
 */
struct groupby_generate_tasks {
  double scale_factor;

  std::vector<std::shared_ptr<gqe::task>> operator()(
    std::vector<std::vector<std::shared_ptr<gqe::task>>> input_tasks,
    gqe::context_reference ctx_ref,
    int32_t& task_id,
    int32_t stage_id)
  {
    assert(input_tasks.size() == 1 && "expected exactly one input relation");
    // For Q18, size the hash map based on lineitem table size
    // Lineitem table: ~6M * scale_factor (TPC-H spec: 6M lineitems per SF)
    // But we're grouping by orderkey, so expect ~1.5M unique orderkeys per SF
    size_t cardinality_estimate = 1'500'000ul * scale_factor;
    auto hash_map               = std::make_shared<task_hash_map>(cardinality_estimate);

    std::vector<std::shared_ptr<gqe::task>> generated_tasks;
    generated_tasks.reserve(input_tasks[0].size());

    for (auto& input_task : input_tasks[0]) {
      generated_tasks.push_back(std::make_shared<groupby_task>(
        ctx_ref, task_id, stage_id, std::move(input_task), hash_map, scale_factor));
      task_id++;
    }

    std::shared_ptr<gqe::task> concatenated_groupby;
    if (generated_tasks.size() == 1) {
      concatenated_groupby = std::move(generated_tasks[0]);
    } else {
      concatenated_groupby =
        std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, generated_tasks);
      task_id++;
    }

    std::shared_ptr<gqe::task> gb_retrieve_task = std::make_shared<groupby_retrieve_task>(
      ctx_ref, task_id, stage_id, concatenated_groupby, hash_map, scale_factor);
    task_id++;

    return {gb_retrieve_task};
  }
};

}  // namespace

std::shared_ptr<gqe::physical::relation> groupby(std::shared_ptr<gqe::physical::relation> lineitem,
                                                 double scale_factor)
{
  std::vector<std::shared_ptr<gqe::physical::relation>> input_wrapper = {std::move(lineitem)};

  groupby_generate_tasks groupby_task_generator{scale_factor};

  // the output of q18's lineitem aggregation is [l_orderkey, sum_l_quantity] which has
  // data types of int64 and float64
  std::vector<cudf::data_type> output_data_types = {cudf::data_type(cudf::type_id::INT64),
                                                    cudf::data_type(cudf::type_id::FLOAT64)};
  return std::make_shared<gqe::physical::user_defined_relation>(
    input_wrapper, groupby_task_generator, false, output_data_types);
}

}  // namespace q18
}  // namespace benchmark
}  // namespace gqe_python

template <typename Identifier>
__global__ void gqe_python::benchmark::q18::groupby_retrieve_kernel(
  gqe_python::utility::q18::map_ref_type<Identifier, double> hash_map_ref,
  cudf::size_type* d_global_offset,
  cudf::mutable_column_device_view out_l_orderkey,
  cudf::mutable_column_device_view out_sum_quantity)
{
  // Shared memory write buffer for coalesced writes
  __shared__ typename utility::write_buffer_op<Identifier, double>::storage_t wbs;

  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_offset_ref(*d_global_offset);

  utility::write_buffer_op<Identifier, double> wb(
    &wbs, global_offset_ref, out_l_orderkey.data<Identifier>(), out_sum_quantity.data<double>());

  auto storage_ref          = hash_map_ref.storage_ref();
  const auto loop_stride    = blockDim.x * gridDim.x;
  const size_t map_capacity = hash_map_ref.capacity();
  constexpr auto bucket_size =
    gqe_python::utility::q18::bucket_size;  // Q18-specific bucket_size = 1
  const auto loop_bound =
    ((map_capacity + bucket_size - 1) / bucket_size + blockDim.x - 1) / blockDim.x * blockDim.x;

  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < loop_bound;
       index += loop_stride) {
    for (size_t i = 0; i < bucket_size; i++) {
      __syncwarp();
      cuda::std::optional<cuda::std::tuple<Identifier, double>> result;

      if (index * bucket_size + i < map_capacity) {
        auto entry = storage_ref[index][i];

        if (entry.first != hash_map_ref.empty_key_sentinel()) {
          auto key   = entry.first;
          auto value = entry.second;

          // Apply TPC-H Q18 filter: sum_quantity > threshold
          if (value > q18_quantity_threshold) { result = cuda::std::make_tuple(key, value); }
        }
      }

      // Buffered write - coalesces memory access and reduces atomic contention
      wb.write(result);
    }
  }

  // Flush any remaining buffered writes
  wb.flush();
}

// Explicit template instantiations
template __global__ void gqe_python::benchmark::q18::groupby_retrieve_kernel<int32_t>(
  gqe_python::utility::q18::map_ref_type<int32_t, double> hash_map_ref,
  cudf::size_type* d_global_offset,
  cudf::mutable_column_device_view out_l_orderkey,
  cudf::mutable_column_device_view out_sum_quantity);

template __global__ void gqe_python::benchmark::q18::groupby_retrieve_kernel<int64_t>(
  gqe_python::utility::q18::map_ref_type<int64_t, double> hash_map_ref,
  cudf::size_type* d_global_offset,
  cudf::mutable_column_device_view out_l_orderkey,
  cudf::mutable_column_device_view out_sum_quantity);
