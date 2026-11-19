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

namespace gqe_bench {

/**
 * @brief Set `NVTX_INJECTION64_PATH` to the NVTX injection this plugin routes through.
 *
 * NVTX reads this env var on its first call (lazy init), `dlopen`s that path, and
 * calls `InitializeInjectionNvtx2` on the target library. Without it, CUPTI's
 * `CUPTI_CB_DOMAIN_NVTX` callbacks never fire.
 *
 * Uses `dladdr` to resolve the path the loader actually bound. An out-of-process
 * resolver (e.g. `ldconfig`) cannot observe the runtime-resolved library and would
 * silently misroute NVTX if the loader's choice ever differed from the cache. A
 * pre-existing value is respected so an outer profiler like `nsys` that owns NVTX
 * routing is not overridden.
 *
 * TODO: the target is this library rather than libcupti as a workaround for a CUPTI
 * defect; see `nvtx_domain_filter.cpp`. To revert once that is fixed, resolve libcupti
 * directly again — `dladdr(&cuptiSubscribe, &info)`, which requires the
 * `<cupti_callbacks.h>` include — and delete `nvtx_domain_filter.cpp`.
 */
void configure_nvtx_injection();

/**
 * @brief Revert a prior `configure_nvtx_injection()` that actually set the env var.
 *
 * `unsetenv`s `NVTX_INJECTION64_PATH` only if this process set it. Pre-existing
 * values (e.g. from an outer profiler) are never touched. Intended for the
 * plugin-construction-failure path so a failed load does not leave the env var
 * orphaned for the rest of the host process. A no-op on success paths —
 * the env var must stay set while the plugin is live.
 */
void unconfigure_nvtx_injection();

}  // namespace gqe_bench
