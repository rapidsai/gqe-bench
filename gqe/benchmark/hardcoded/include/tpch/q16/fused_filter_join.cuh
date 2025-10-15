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

#pragma once

#include <tpch/q16/task.hpp>
#include <utility/config.hpp>
#include <utility/write_buffer.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>

namespace gqe_python {
namespace benchmark {
namespace q16 {

template <typename T>
using part_map_type = gqe_python::utility::map_type<T, cudf::size_type>;
template <typename T>
using part_map_ref_type = typename part_map_type<
  T>::ref_type<cuco::op::find_tag, cuco::op::for_each_tag, cuco::op::insert_tag>;
template <typename T>
using supplier_map_type = gqe_python::utility::map_type<T, cudf::size_type>;
template <typename T>
using supplier_map_ref_type = typename supplier_map_type<
  T>::ref_type<cuco::op::find_tag, cuco::op::for_each_tag, cuco::op::insert_tag>;

// Keys are non-negative in nature.
template <typename T>
constexpr cuco::empty_key<T> empty_key_sentinel(-1);
// Row indices are non-negative in nature.
constexpr cuco::empty_value<cudf::size_type> empty_value_sentinel(-1);

/**
 * @brief Fused kernel for part table filter and hash table build.
 *
 * @tparam Identifier The type of the identifier (int32_t, int64_t).
 * @tparam EnableBloomFilter Whether to enable bloom filter for part hash table.
 *
 * @param[in] p_partkey_column part table p_partkey column.
 * @param[in] p_brand_column part table p_brand column.
 * @param[in] p_type_column part table p_type column.
 * @param[in] p_size_column part table p_size column.
 * @param[in, out] part_map hash map used for part table.
 * @param[in, out] bloom_filter bloom filter used for part table.
 */
template <typename Identifier, bool EnableBloomFilter = false>
__global__ void fused_filter_hashtable_build_kernel(
  cudf::column_device_view p_partkey_column,
  cudf::column_device_view p_brand_column,
  cudf::column_device_view p_type_column,
  cudf::column_device_view p_size_column,
  part_map_ref_type<Identifier> part_map,
  gqe_python::utility::bloom_filter_ref_type<Identifier> bloom_filter);

/**
 * @brief The parameters used for fused join kernel. They are wrapped in a struct because nvcc will
 * throw compile errors for long symbols (cuco::multimap and cuco::bloom_filter would have long
 * symbols).
 *
 * @tparam Identifier The type of the identifier (int32_t, int64_t).
 */
template <typename Identifier>
struct fused_join_params {
  cudf::column_device_view ps_partkey_column;      ///< partsupp partkey column
  cudf::column_device_view ps_suppkey_column;      ///< partsupp suppkey column
  supplier_map_ref_type<Identifier> supplier_map;  ///< supplier hash map
  gqe_python::utility::bloom_filter_ref_type<Identifier>
    supplier_bloom_filter;                 ///< supplier bloom filter
  part_map_ref_type<Identifier> part_map;  ///< part hash map
  gqe_python::utility::bloom_filter_ref_type<Identifier> part_bloom_filter;  ///< part bloom filter
  cudf::size_type* p_out_indices;   ///< output part indices
  cudf::size_type* ps_out_indices;  ///< output partsupp indices
  cudf::size_type* d_counter;       ///< global counter for output rows
};

/**
 * @brief Fused kernel for partsupp table probing part hash table and supplier hash table.
 *
 * @tparam Identifier The type of the identifier (int32_t, int64_t).
 * @param[in, out] params params used for the fused join kernel.
 */
template <typename Identifier,
          bool EnablePartBloomFilter     = false,
          bool EnableSupplierBloomFilter = false>
__global__ void fused_join_kernel(fused_join_params<Identifier> params);

}  // namespace q16
}  // namespace benchmark
}  // namespace gqe_python
