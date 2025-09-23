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

#include <memory>

namespace gqe_python {
namespace benchmark {
namespace q13 {

/**
 * @brief Compute the fused project filter relation.
 *
 * @param[in] input Input relation.
 */
std::shared_ptr<gqe::physical::relation> filter_orders(
  std::shared_ptr<gqe::physical::relation> input);

}  // namespace q13
}  // namespace benchmark
}  // namespace gqe_python
