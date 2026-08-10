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

// Trivial CUDA kernel for the CUPTI wrapper tests. Purpose: produce real SM
// work so the profilers have something to measure. The kernel computes
// nothing meaningful — it's a MAC loop per thread.

#include "cuda_error.hpp"

#include <cuda_runtime.h>

namespace {

constexpr int k_n       = 1024;
constexpr int k_threads = 256;
constexpr int k_blocks  = (k_n + k_threads - 1) / k_threads;

__global__ void mac_kernel(float* out, int n)
{
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  float s = 0.0f;
  // Mix `tid` into the accumulator so the inner loop is not a pure function of
  // constants. Without it every thread computes the same final value and a
  // future nvcc could constant-fold the loop, eliding the SM work the CUPTI
  // tests exist to measure.
  for (int i = 0; i < 256; ++i) {
    s = s * 1.0001f + static_cast<float>(i ^ tid);
  }
  out[tid] = s;
}

}  // namespace

// Allocate the kernel's output buffer. Caller owns, frees via
// `free_test_kernel_out`. Throws `gqe_bench::cuda_error` on failure.
extern "C" float* allocate_test_kernel_out()
{
  float* d_out = nullptr;
  GQE_BENCH_CUDA_TRY(cudaMalloc(&d_out, k_n * sizeof(float)));
  return d_out;
}

extern "C" void free_test_kernel_out(float* d_out)
{
  if (d_out != nullptr) { GQE_BENCH_CUDA_TRY(cudaFree(d_out)); }
}

// Launch the kernel on `stream`, writing into `d_out`. Checks the async
// launch error immediately so a bad configuration fails loudly instead of
// producing a silent no-op run.
extern "C" void launch_test_kernel(cudaStream_t stream, float* d_out)
{
  mac_kernel<<<k_blocks, k_threads, 0, stream>>>(d_out, k_n);
  GQE_BENCH_CUDA_TRY(cudaGetLastError());
}
