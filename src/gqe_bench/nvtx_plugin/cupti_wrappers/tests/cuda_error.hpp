/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

// Minimal CUDA error-check macro for the plugin's standalone tests; lets
// the test harness link without `libgqe`. Used by `cupti_test.cpp` and
// `test_kernel.cu`.

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace gqe_bench {

/**
 * @brief Exception type thrown by `GQE_BENCH_CUDA_TRY` on a non-success CUDA result.
 */
struct cuda_error : public std::runtime_error {
  cuda_error(char const* message) : std::runtime_error(message) {}
  cuda_error(std::string const& message) : cuda_error{message.c_str()} {}
};

}  // namespace gqe_bench

#define GQE_BENCH_STRINGIFY_DETAIL(x) #x
#define GQE_BENCH_STRINGIFY(x)        GQE_BENCH_STRINGIFY_DETAIL(x)

/**
 * @brief Throw `gqe_bench::cuda_error` if the CUDA call returns non-success.
 *
 * The thrown `what()` carries file, line, `cudaGetErrorName`, and
 * `cudaGetErrorString` so the gtest failure points at the real call site.
 * Swallows any pending error via `cudaGetLastError()` to avoid sticky errors
 * leaking across tests.
 */
#define GQE_BENCH_CUDA_TRY(_call)                                                                 \
  do {                                                                                            \
    cudaError_t const _err = (_call);                                                             \
    if (cudaSuccess != _err) {                                                                    \
      cudaGetLastError();                                                                         \
      throw gqe_bench::cuda_error{std::string{"CUDA error at: "} + __FILE__ + ":" +               \
                                  GQE_BENCH_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(_err) + \
                                  " " + cudaGetErrorString(_err)};                                \
    }                                                                                             \
  } while (0)

namespace gqe_bench {

/**
 * @brief Return the CUDA device id the calling thread is bound to.
 *
 * Preferred over hardcoding `0` so tests honour `CUDA_VISIBLE_DEVICES`.
 * Throws `gqe_bench::cuda_error` on failure.
 */
inline int current_cuda_device()
{
  int id{};
  GQE_BENCH_CUDA_TRY(cudaGetDevice(&id));
  return id;
}

}  // namespace gqe_bench
