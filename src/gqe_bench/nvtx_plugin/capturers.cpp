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

#include "capturers.hpp"

#include <cuda_runtime_api.h>

#include <memory>
#include <utility>

namespace gqe_bench {

void activity_capturer::start()
{
  if (_state != capturer_state::armed) return;
  if (!_profiler) { _profiler = std::make_unique<cupti::activity_profiler>(); }
  _profiler->start();
  _state = capturer_state::running;
}

void activity_capturer::stop(run_context& ctx)
{
  if (_state != capturer_state::running) return;
  auto records         = _profiler->stop();
  ctx.breakdown        = cupti::activity_profiler::get_time_breakdown(records);
  ctx.breakdown_valid  = true;
  ctx.activity_records = std::move(records);
  _state               = capturer_state::armed;
}

void counter_capturer::start()
{
  if (_state != capturer_state::armed) return;
  if (!_profiler) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    _profiler = std::make_unique<cupti::user_range_profiler>(
      cupti::user_range_profiler::configuration{dev, _metrics});
  }
  _profiler->start();
  _state = capturer_state::running;
}

void counter_capturer::stop(run_context& ctx)
{
  if (_state != capturer_state::running) return;
  auto profile       = _profiler->stop();
  ctx.counter_values = std::move(profile.metric_values);
  _state             = capturer_state::armed;
}

}  // namespace gqe_bench
