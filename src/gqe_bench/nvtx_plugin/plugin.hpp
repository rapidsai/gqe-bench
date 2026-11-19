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

#include "observers.hpp"

#include <cupti_callbacks.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace gqe_bench {

/**
 * @brief The plugin's resolved environment, read once at initialization.
 *
 * Holding one is the statement that the plugin has work to do: it is only
 * produced when both the database path and a usable shared-memory segment
 * are configured. The two feature fields are independent of that — with
 * both off the plugin still records `run` rows and stage timings.
 */
struct plugin_config {
  /**
   * @brief Path to the SQLite database rank 0 writes to.
   */
  std::string db_path;

  /**
   * @brief Segment the `execute_plan_observer` attaches to.
   */
  shm_config shm;

  /**
   * @brief Whether to capture CUPTI activity records.
   */
  bool time_breakdown = false;

  /**
   * @brief CUPTI Range Profiler metrics to collect; empty disables.
   */
  std::vector<std::string> cupti_metrics;
};

/**
 * @brief Owns the observer registry and the CUPTI subscription.
 *
 * Resource lifetimes for the database connection and shared-memory
 * attach are owned by the observers; the plugin holds no per-run
 * state.
 */
class plugin {
 public:
  /**
   * @brief Take ownership of the observers, register the CUPTI subscription,
   * and enable the NVTX callback domain.
   *
   * @param[in] observers Registry to own, built by the caller from a
   *                      `plugin_config`. Contents are not validated: the
   *                      subscription is enabled either way, and a range
   *                      with no observer in the registry is silently
   *                      unobserved.
   * @throw std::runtime_error On CUPTI subscribe or enable failure.
   *                           `_subscriber` leaks if subscribe succeeds
   *                           then enable fails.
   */
  explicit plugin(std::vector<std::unique_ptr<range_observer>> observers);

  /**
   * @brief Disable the CUPTI subscription if active. Observers are
   * destroyed in reverse declaration order.
   */
  ~plugin();

  plugin(plugin const&)            = delete;
  plugin& operator=(plugin const&) = delete;

  /**
   * @brief Look up an observer by NVTX range name.
   *
   * @return The matching observer pointer, or null if `name` is null or
   *         no observer claims it.
   */
  range_observer* find_observer(char const* name);

 private:
  /**
   * @brief Observer registry: the outer `execute_plan` observer plus one
   * per pipeline stage, looked up by NVTX range name via `find_observer`.
   */
  std::vector<std::unique_ptr<range_observer>> _observers;

  /**
   * @brief Active CUPTI subscription, or null.
   */
  CUpti_SubscriberHandle _subscriber = nullptr;
};

/**
 * @brief The live plugin. Read concurrently by CUPTI callback threads;
 * written by `disable_plugin()` (itself called from a callback's
 * exception handler).
 *
 * A callback loads its own `shared_ptr` copy on entry so the plugin
 * cannot be destroyed underneath an in-flight callback when another
 * callback stores `nullptr`.
 *
 * `std::atomic<std::shared_ptr<T>>` is the C++20 specialization providing
 * atomic load/store/exchange of the shared pointer.
 */
extern std::atomic<std::shared_ptr<plugin>> g_plugin;

/**
 * @brief Drop the global plugin singleton.
 *
 * After this call, `g_plugin.load()` returns null and subsequent callbacks
 * short-circuit. Existing in-flight callbacks finish via their own
 * `shared_ptr` copies; the last one triggers `~plugin()`.
 */
void disable_plugin() noexcept;

}  // namespace gqe_bench
