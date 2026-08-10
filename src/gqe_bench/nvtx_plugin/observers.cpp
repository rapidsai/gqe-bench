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

#include "observers.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <memory>
#include <string>
#include <utility>

namespace gqe_bench {

thread_local run_context t_run_context;

execute_plan_observer::execute_plan_observer(std::string db_path, shm_config shm_cfg)
  : _db_path(std::move(db_path)), _shm_cfg(std::move(shm_cfg))
{
}

void execute_plan_observer::attach()
{
  if (_attached) return;
  _attached = true;

  int dev              = -1;
  cudaError_t const rc = cudaGetDevice(&dev);
  if (rc != cudaSuccess || dev < 0) {
    throw std::runtime_error(std::format(
      "execute_plan_observer::attach: cudaGetDevice rc={} dev={}", static_cast<int>(rc), dev));
  }

  _shm = agg_protocol::attach(
    _shm_cfg.name.c_str(), _shm_cfg.size, _shm_cfg.total_ranks, static_cast<std::uint32_t>(dev));
  _shm->barrier();
}

void execute_plan_observer::on_push(CUpti_NvtxData const* /*cbdata*/)
{
  if (!_attached) attach();

  std::int32_t const requested = _shm->requested_experiment_id();
  if (requested == 0) return;

  auto& ctx = t_run_context;
  ctx.reset();
  ctx.in_execute_plan = true;
  _shm->reset_run_state();

  if (_shm->is_rank_zero()) {
    if (!_writer.has_value()) { _writer.emplace(_db_path, _shm->total_ranks()); }
    ctx.experiment_id = requested;
  }
}

void execute_plan_observer::on_pop()
{
  auto& ctx = t_run_context;
  if (!ctx.in_execute_plan) return;
  ctx.in_execute_plan = false;

  // Every rank publishes its own contribution. Rank 0 (the only rank with
  // ``_writer``) then gathers all ranks and commits.
  _shm->publish_self(ctx);
  if (!_writer.has_value() || !ctx.experiment_id) return;
  std::int64_t const exp_id = *ctx.experiment_id;
  ctx.experiment_id.reset();
  _writer->write_run(_shm->gather_run(), exp_id);
}

stage_observer::stage_observer(stage s) : _stage(s) {}

void stage_observer::on_push(CUpti_NvtxData const*)
{
  auto& ctx = t_run_context;
  if (!ctx.in_execute_plan) return;
  ctx.stage_starts[static_cast<std::size_t>(_stage)] = std::chrono::steady_clock::now();
}

void stage_observer::on_pop()
{
  auto& ctx = t_run_context;
  if (!ctx.in_execute_plan) return;
  auto i                 = static_cast<std::size_t>(_stage);
  ctx.stage_durations[i] = std::chrono::steady_clock::now() - ctx.stage_starts[i];
}

execute_stage_observer::execute_stage_observer(bool time_breakdown,
                                               std::vector<std::string> const& cupti_metrics)
{
  if (time_breakdown) { _capturers.push_back(std::make_unique<activity_capturer>()); }
  if (!cupti_metrics.empty()) {
    _capturers.push_back(std::make_unique<counter_capturer>(cupti_metrics));
  }
}

void execute_stage_observer::on_push(CUpti_NvtxData const*)
{
  auto& ctx = t_run_context;
  if (!ctx.in_execute_plan) return;
  record_stage_start();
  for (auto& c : _capturers)
    c->start();
}

void execute_stage_observer::on_pop()
{
  auto& ctx = t_run_context;
  if (!ctx.in_execute_plan) return;
  record_stage_end();
  for (auto& c : _capturers)
    c->stop(ctx);
}

void execute_stage_observer::record_stage_start() noexcept
{
  t_run_context.stage_starts[static_cast<std::size_t>(stage::execute)] =
    std::chrono::steady_clock::now();
}

void execute_stage_observer::record_stage_end() noexcept
{
  auto i = static_cast<std::size_t>(stage::execute);
  t_run_context.stage_durations[i] =
    std::chrono::steady_clock::now() - t_run_context.stage_starts[i];
}

}  // namespace gqe_bench
