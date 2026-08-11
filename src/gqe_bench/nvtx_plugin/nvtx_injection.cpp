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

#include "nvtx_injection.hpp"

#include "log.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <cstdlib>

namespace gqe_bench {

namespace {

// True only if the successful `setenv` call below was made by this process.
// A pre-existing value (respected path) does NOT flip this, so
// `unconfigure_nvtx_injection()` never unsets something we did not set.
bool g_nvtx_injection_set_by_us = false;

}  // namespace

void configure_nvtx_injection()
{
  // TODO: workaround. NVTX is pointed at this library, whose
  // `nvtx_domain_filter.cpp` chains to libcupti and drops out-of-domain events, rather
  // than at libcupti directly. To revert when the underlying CUPTI defect is fixed,
  // take the address of `cuptiSubscribe` here instead (restoring the
  // `<cupti_callbacks.h>` include) so `info.dli_fname` names libcupti again.
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void const*>(&configure_nvtx_injection), &info) || !info.dli_fname) {
    GQE_BENCH_LOG_ERROR(
      "dladdr(configure_nvtx_injection) failed; NVTX->CUPTI routing may not activate pid={}",
      getpid());
    return;
  }
  char const* existing = std::getenv("NVTX_INJECTION64_PATH");
  if (existing && *existing) {
    GQE_BENCH_LOG_INFO(
      "NVTX_INJECTION64_PATH already set to {} (respected) pid={}", existing, getpid());
    return;
  }
  if (setenv("NVTX_INJECTION64_PATH", info.dli_fname, 0) == 0) {
    g_nvtx_injection_set_by_us = true;
    GQE_BENCH_LOG_INFO("NVTX_INJECTION64_PATH auto-set to {} pid={}", info.dli_fname, getpid());
  } else {
    GQE_BENCH_LOG_ERROR("setenv NVTX_INJECTION64_PATH failed pid={}", getpid());
  }
}

void unconfigure_nvtx_injection()
{
  if (!g_nvtx_injection_set_by_us) return;
  if (unsetenv("NVTX_INJECTION64_PATH") == 0) {
    g_nvtx_injection_set_by_us = false;
    GQE_BENCH_LOG_INFO("NVTX_INJECTION64_PATH unset (plugin construction failed) pid={}", getpid());
  } else {
    GQE_BENCH_LOG_ERROR("unsetenv NVTX_INJECTION64_PATH failed pid={}", getpid());
  }
}

}  // namespace gqe_bench
