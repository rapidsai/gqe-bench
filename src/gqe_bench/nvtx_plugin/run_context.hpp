/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include "cupti_wrappers/cupti_activity.hpp"
#include <nvtx_plugin/stages.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace gqe_bench {

/**
 * @brief Per-thread state populated by the observers during one
 * `execute_plan` range. Observer instances are process-global, so
 * per-range state cannot live on observer members.
 */
struct run_context {
  /**
   * @brief True between the outer push and pop on every rank.
   */
  bool in_execute_plan = false;

  /**
   * @brief Populated on rank 0 only; empty elsewhere.
   */
  std::optional<int64_t> experiment_id;

  /**
   * @brief Per-stage push timestamp.
   */
  std::array<std::chrono::steady_clock::time_point, stages.size()> stage_starts{};

  /**
   * @brief Per-stage `pop - push` duration.
   */
  std::array<std::chrono::steady_clock::duration, stages.size()> stage_durations{};

  /**
   * @brief True iff `breakdown` was populated by a CUPTI Activity stop.
   */
  bool breakdown_valid = false;

  /**
   * @brief Derived per-stage time breakdown (rank-local).
   */
  cupti::time_breakdown breakdown{};

  /**
   * @brief Raw CUPTI events for the range; forwarded to rank 0 for per-event rows.
   *
   * Moved out by `agg_protocol::publish_self`; re-initialized by
   * `reset()` at the next `execute_plan` push.
   */
  cupti::activity_records activity_records{};

  /**
   * @brief CUPTI Range Profiler counter values keyed by metric name.
   */
  std::unordered_map<std::string, double> counter_values{};

  /**
   * @brief Clear per-range state. Callers set `in_execute_plan` and may
   * populate `experiment_id` afterward.
   */
  void reset() noexcept
  {
    in_execute_plan = false;
    experiment_id.reset();
    stage_starts.fill({});
    stage_durations.fill({});
    breakdown_valid  = false;
    breakdown        = {};
    activity_records = {};
    counter_values.clear();
  }
};

extern thread_local run_context t_run_context;

}  // namespace gqe_bench
