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
#include "cupti_wrappers/cupti_range.hpp"
#include "run_context.hpp"

#include <memory>
#include <string>
#include <vector>

namespace gqe_bench {

/**
 * @brief State of an opt-in CUPTI capturer.
 */
enum class capturer_state {
  /**
   * @brief Not opted in by configuration; all paths are no-ops.
   */
  disabled,
  /**
   * @brief Opted in; no live CUPTI session. Next `start()` begins one.
   */
  armed,
  /**
   * @brief In a measurement window. Next `stop()` ends it.
   */
  running,
};

/**
 * @brief Strategy interface for an opt-in CUPTI capturer.
 *
 * Implementations own their `capturer_state` and write results into a
 * `run_context` on `stop`. Failures throw.
 */
class capturer {
 public:
  virtual ~capturer() = default;

  /**
   * @brief Begin a measurement window if armed.
   */
  virtual void start() = 0;

  /**
   * @brief End a measurement window and publish results into `ctx`.
   */
  virtual void stop(run_context& ctx) = 0;

  /**
   * @brief Short identifier used in log lines.
   */
  virtual char const* name() const = 0;
};

/**
 * @brief Capturer for the CUPTI Activity API. Constructs the underlying
 * `cupti::activity_profiler` lazily on first `start` because the CUDA
 * context is not live at plugin load. Always starts armed.
 */
class activity_capturer final : public capturer {
 public:
  activity_capturer() : _state(capturer_state::armed) {}

  void start() override;
  void stop(run_context& ctx) override;
  char const* name() const override { return "activity"; }

 private:
  /**
   * @brief Constructed lazily on first `start`.
   */
  std::unique_ptr<cupti::activity_profiler> _profiler;

  /**
   * @brief Current capturer state.
   */
  capturer_state _state;
};

/**
 * @brief Capturer for the CUPTI Range Profiler. Starts disabled when no
 * metrics are configured and armed otherwise. The underlying
 * `cupti::user_range_profiler` is constructed lazily on first `start`.
 */
class counter_capturer final : public capturer {
 public:
  /**
   * @brief Construct with the list of metric names to profile. An empty
   * list leaves the capturer permanently disabled.
   */
  explicit counter_capturer(std::vector<std::string> metrics)
    : _metrics(std::move(metrics)),
      _state(_metrics.empty() ? capturer_state::disabled : capturer_state::armed)
  {
  }

  void start() override;
  void stop(run_context& ctx) override;
  char const* name() const override { return "counter"; }

 private:
  /**
   * @brief Metric names requested at construction.
   */
  std::vector<std::string> _metrics;

  /**
   * @brief Constructed lazily on first `start`.
   */
  std::unique_ptr<cupti::user_range_profiler> _profiler;

  /**
   * @brief Current capturer state.
   */
  capturer_state _state;
};

}  // namespace gqe_bench
