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

#include "agg_protocol.hpp"
#include "capturers.hpp"
#include "run_context.hpp"
#include "run_writer.hpp"
#include <nvtx_plugin/stages.hpp>

#include <cupti_callbacks.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gqe_bench {

/**
 * @brief Configuration values for the shared-memory attach, carried in
 * `plugin_config` from the environment read into `execute_plan_observer`.
 */
struct shm_config {
  /**
   * @brief POSIX segment name with leading '/'.
   */
  std::string name;

  /**
   * @brief Number of ranks participating in the segment.
   */
  std::uint32_t total_ranks = 0;

  /**
   * @brief Total segment size in bytes.
   */
  std::size_t size = 0;
};

/**
 * @brief Interface for a per-NVTX-range observer.
 *
 * Implementations are invoked from a CUPTI callback. Overrides may throw;
 * exceptions are caught at the `cupti_callback` boundary.
 */
struct range_observer {
  virtual ~range_observer() = default;

  /**
   * @brief NVTX range name this observer claims.
   */
  virtual char const* name() const = 0;

  /**
   * @brief Called when the observer's range is pushed.
   */
  virtual void on_push(CUpti_NvtxData const* cbdata) = 0;

  /**
   * @brief Called when the observer's range is popped.
   */
  virtual void on_pop() = 0;
};

/**
 * @brief Observer for the outer `execute_plan` range.
 *
 * On the first push, resolves the rank via `cudaGetDevice`, attaches to
 * the shared-memory segment, crosses the attach barrier, and (on rank 0)
 * constructs a `run_writer` for database commits. On every push, resets
 * per-run state and (on rank 0) reads the active experiment id. On pop,
 * publishes this rank's contribution and (on rank 0) gathers all ranks
 * and writes one transaction.
 */
class execute_plan_observer final : public range_observer {
 public:
  /**
   * @brief Construct with the database path and the shm configuration.
   * Heavy work (DB open, shm attach, barrier) is deferred until
   * `attach` runs on the first observed push.
   */
  execute_plan_observer(std::string db_path, shm_config shm_cfg);
  ~execute_plan_observer() override = default;

  execute_plan_observer(execute_plan_observer const&)            = delete;
  execute_plan_observer& operator=(execute_plan_observer const&) = delete;

  char const* name() const override { return k_execute_plan_range; }
  void on_push(CUpti_NvtxData const* cbdata) override;
  void on_pop() override;

 private:
  /**
   * @brief Resolve the rank, attach to the segment, run the barrier, and
   * (on rank 0) construct the `run_writer`. Idempotent on success: once
   * `_attached` is true subsequent calls are no-ops.
   *
   * @throw std::runtime_error If `cudaGetDevice` fails, `agg_protocol::attach`
   *        fails, or (on rank 0) `run_writer` construction fails.
   */
  void attach();

  /**
   * @brief Path passed to the future `run_writer`.
   */
  std::string _db_path;

  /**
   * @brief Shm configuration captured at construction.
   */
  shm_config _shm_cfg;

  /**
   * @brief True after a successful `attach`.
   */
  bool _attached = false;

  /**
   * @brief Set after a successful attach.
   */
  std::optional<agg_protocol> _shm;

  /**
   * @brief Rank-0-only owner of the database connection and prepared
   * statements. `_writer.has_value()` is the authoritative check for
   * "this process writes to the DB".
   */
  std::optional<run_writer> _writer;
};

/**
 * @brief Observer for a pure-timing stage (`build_task_graph` or
 * `collect_results`). Records the stage's start timestamp on push and
 * computes its duration on pop. No-op outside of an active
 * `execute_plan` range.
 */
class stage_observer final : public range_observer {
 public:
  /**
   * @brief Construct an observer for stage `s`.
   */
  explicit stage_observer(stage s);

  char const* name() const override { return info_for(_stage).nvtx_range_name; }
  void on_push(CUpti_NvtxData const* cbdata) override;
  void on_pop() override;

 private:
  /**
   * @brief Stage this observer represents.
   */
  stage _stage;
};

/**
 * @brief Observer for the `execute_task_graph` range. Records the
 * execute-stage timestamps and delegates `start` / `stop` to its
 * configured capturers.
 */
class execute_stage_observer final : public range_observer {
 public:
  /**
   * @brief Construct with the capture toggles. Installs an
   * `activity_capturer` only if `time_breakdown` is set, and a
   * `counter_capturer` only if `cupti_metrics` is non-empty. With both
   * off the observer records stage timing and nothing else, and no
   * CUPTI profiler is ever constructed.
   */
  execute_stage_observer(bool time_breakdown, std::vector<std::string> const& cupti_metrics);

  char const* name() const override { return info_for(stage::execute).nvtx_range_name; }
  void on_push(CUpti_NvtxData const* cbdata) override;
  void on_pop() override;

 private:
  /**
   * @brief Record the execute-stage push timestamp into the thread-local
   * `run_context`.
   */
  void record_stage_start() noexcept;

  /**
   * @brief Compute the execute-stage duration into the thread-local
   * `run_context`.
   */
  void record_stage_end() noexcept;

  /**
   * @brief Capturers in start/stop order.
   */
  std::vector<std::unique_ptr<capturer>> _capturers;
};

}  // namespace gqe_bench
