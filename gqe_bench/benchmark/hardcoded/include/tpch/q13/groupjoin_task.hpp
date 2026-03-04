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
namespace q13 {

/**
 * @brief Create a groupjoin hash map build relation.
 *
 * @param[in] customer The LHS input table of the groupjoin. In Q13, this is the `customer` table.
 * @param[in] scale_factor The TPC-H scale factor. The SF used to calculate the size of the
 * `customer` table. This is a workaround for not having access to the GQE catalog.
 *
 * @return The groupjoin build physical relation.
 */
std::shared_ptr<gqe::physical::relation> groupjoin_build(
  std::shared_ptr<gqe::physical::relation> customer, double scale_factor);

/**
 * @brief Create a groupjoin hash map probe relation.
 *
 * @param[in] groupjoin_build The build-side relation of the groupjoin.
 * @param[in] orders The filtered orders input relation.
 *
 * @return The groupjoin probe physical relation.
 */
std::shared_ptr<gqe::physical::relation> groupjoin_probe(
  std::shared_ptr<gqe::physical::relation> groupjoin_build,
  std::shared_ptr<gqe::physical::relation> orders);

/**
 * @brief A fused relation for the filter and groupjoin probe.
 *
 * @param[in] groupjoin_build The build relation of the groupjoin.
 * @param[in] orders The unfiltered orders input relation.
 *
 * @return The fused physical relation.
 */
std::shared_ptr<gqe::physical::relation> fused_filter_probe(
  std::shared_ptr<gqe::physical::relation> groupjoin_build,
  std::shared_ptr<gqe::physical::relation> orders);

/**
 * @brief Create a groupjoin hash map retrieve relation.
 *
 * @param[in] groupjoin_probe The probe relation of the groupjoin.
 *
 * @return The groupjoin retrieve physical relation.
 */
std::shared_ptr<gqe::physical::relation> groupjoin_retrieve(
  std::shared_ptr<gqe::physical::relation> groupjoin_probe);

}  // namespace q13
}  // namespace benchmark
}  // namespace gqe_python
