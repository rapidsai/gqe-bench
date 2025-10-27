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
                                                 int32_t scale_factor);

}  // namespace q18
}  // namespace benchmark
}  // namespace gqe_python
