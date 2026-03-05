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

#include <gqe/physical/relation.hpp>

#include <cstdint>
#include <memory>

namespace gqe_python {
namespace benchmark {
namespace q18 {

/**
 * @brief Creates a group by relation for Q18.
 *
 * This performs aggregation on lineitem table, grouping by l_orderkey
 * and summing l_quantity, with atomic initialization of buckets.
 *
 * @param lineitem The lineitem relation with [l_orderkey, l_quantity]
 * @param scale_factor The TPC-H scale factor for sizing estimates
 * @return A group by relation with [l_orderkey, sum_l_quantity] where sum_l_quantity > 300
 */
std::shared_ptr<gqe::physical::relation> groupby(std::shared_ptr<gqe::physical::relation> lineitem,
                                                 double scale_factor);

}  // namespace q18
}  // namespace benchmark
}  // namespace gqe_python
