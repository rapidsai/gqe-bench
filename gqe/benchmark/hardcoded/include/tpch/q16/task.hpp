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

#include <gqe/executor/concatenate.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/for_each.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>

namespace gqe_python {
namespace benchmark {
namespace q16 {

/**
 * @brief Construct the aggregate relation for COUNT(DISTINCT).
 *
 * @param[in] input_table
 */
std::shared_ptr<gqe::physical::relation> aggregate(
  std::shared_ptr<gqe::physical::relation> input_table);

/**
 * @brief Construct the fused filter and join relation.
 *
 * @param[in] supplier_table supplier table relation.
 * @param[in] part_table part table relation.
 * @param[in] partsupp_table partsupp table relation
 */
std::shared_ptr<gqe::physical::relation> fused_filter_join(
  std::shared_ptr<gqe::physical::relation> supplier_table,
  std::shared_ptr<gqe::physical::relation> part_table,
  std::shared_ptr<gqe::physical::relation> partsupp_table);

}  // namespace q16
}  // namespace benchmark
}  // namespace gqe_python
