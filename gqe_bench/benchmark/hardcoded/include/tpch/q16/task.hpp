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
