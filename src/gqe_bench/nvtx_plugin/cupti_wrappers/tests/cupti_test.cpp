/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Standalone gtest harness for the CUPTI wrappers. The GPU workload is a
// raw CUDA kernel (test_kernel.cu) on a plain cudaStream_t so the test
// binary has no heavy dependencies.

#include "../cupti_activity.hpp"
#include "../cupti_range.hpp"
#include "cuda_error.hpp"

#include <cuda_runtime.h>
#include <cupti_callbacks.h>
#include <dlfcn.h>
#include <nvtx3/nvToolsExt.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// Defined in test_kernel.cu — launches a trivial MAC kernel on `stream`,
// writing into the caller-owned output buffer. Throws
// `gqe_bench::cuda_error` on failure.
extern "C" float* allocate_test_kernel_out();
extern "C" void free_test_kernel_out(float* d_out);
extern "C" void launch_test_kernel(cudaStream_t stream, float* d_out);

namespace {

// Minimal RAII NVTX range for these standalone tests.
class nvtx_scoped_range {
 public:
  explicit nvtx_scoped_range(char const* name) { nvtxRangePushA(name); }
  ~nvtx_scoped_range() { nvtxRangePop(); }
  nvtx_scoped_range(nvtx_scoped_range const&)            = delete;
  nvtx_scoped_range& operator=(nvtx_scoped_range const&) = delete;
};

// Routes NVTX calls through CUPTI's activity API by setting
// NVTX_INJECTION64_PATH to libcupti.so before any NVTX call. Mirrors the
// plugin's `configure_nvtx_injection()` (nvtx_injection.cpp) for standalone
// tests. NVTX reads this env var lazily on first call, so it's sufficient to
// set it before any test that uses NVTX fires.
//
// On dladdr failure we log and return — matching the plugin's behavior. Any
// test that depends on NVTX->CUPTI routing (e.g. NvtxMarkerCaptured) will fail
// with an informative symptom rather than being silently skipped.
class NvtxInjectionEnvironment : public ::testing::Environment {
 public:
  void SetUp() override
  {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void const*>(&cuptiSubscribe), &info) || !info.dli_fname) {
      std::fprintf(stderr,
                   "[cupti_test] dladdr(cuptiSubscribe) failed; NVTX->CUPTI routing may not "
                   "activate. Tests depending on NVTX markers will fail.\n");
      return;
    }
    char const* existing = std::getenv("NVTX_INJECTION64_PATH");
    if (existing && *existing) {
      std::fprintf(
        stderr, "[cupti_test] NVTX_INJECTION64_PATH already set to %s (respected)\n", existing);
      return;
    }
    if (setenv("NVTX_INJECTION64_PATH", info.dli_fname, /*overwrite=*/0) != 0) {
      std::fprintf(stderr, "[cupti_test] setenv NVTX_INJECTION64_PATH failed\n");
    }
  }
};

[[maybe_unused]] ::testing::Environment* const kNvtxEnv =
  ::testing::AddGlobalTestEnvironment(new NvtxInjectionEnvironment);

}  // namespace

/**
 * @brief Base fixture shared by UserRangeProfilerTest and ActivityProfilerTest.
 * Owns a CUDA stream and provides two workload helpers.
 */
class CuptiTestBase : public ::testing::Test {
 public:
  void SetUp() override
  {
    // TODO: migrate to CCCL cuda::stream (CUDA 13.2) for RMM-free stream/device/buffer wrappers.
    GQE_BENCH_CUDA_TRY(cudaStreamCreate(&stream));
    d_out = allocate_test_kernel_out();
  }

  void TearDown() override
  {
    free_test_kernel_out(d_out);
    d_out = nullptr;
    GQE_BENCH_CUDA_TRY(cudaStreamDestroy(stream));
  }

  /// Launches a trivial kernel on the stream and synchronizes.
  void run_cuda_fn()
  {
    launch_test_kernel(stream, d_out);
    GQE_BENCH_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  /// Small host-to-device memcpy, for memcpy-activity tests.
  void run_memcpy()
  {
    std::vector<uint8_t> host_buf(memcpy_num_bytes, 0);
    void* device_buf = nullptr;
    GQE_BENCH_CUDA_TRY(cudaMalloc(&device_buf, memcpy_num_bytes));
    GQE_BENCH_CUDA_TRY(
      cudaMemcpy(device_buf, host_buf.data(), memcpy_num_bytes, cudaMemcpyDefault));
    GQE_BENCH_CUDA_TRY(cudaFree(device_buf));
  }

  static constexpr size_t memcpy_num_bytes = 1024;

  cudaStream_t stream{};
  float* d_out{nullptr};
};

class UserRangeProfilerTest : public CuptiTestBase {
 public:
  static constexpr auto metric = "sm__inst_executed.sum";
};

/**
 * @brief A simple use of the profiler.
 */
TEST_F(UserRangeProfilerTest, Simple)
{
  gqe_bench::cupti::user_range_profiler::configuration config;
  config.device_id = gqe_bench::current_cuda_device();
  config.metrics   = {metric};

  gqe_bench::cupti::user_range_profiler profiler(config);
  profiler.start();
  run_cuda_fn();
  auto profile = profiler.stop();

  EXPECT_TRUE(profile.metric_values.contains(metric));
  EXPECT_GT(profile.metric_values[metric], 0);
}

/**
 * @brief Use the profiler multiple times between setup and teardown.
 */
TEST_F(UserRangeProfilerTest, MultiUse)
{
  int32_t constexpr runs = 5;

  gqe_bench::cupti::user_range_profiler::configuration config;
  config.device_id = gqe_bench::current_cuda_device();
  config.metrics   = {metric};

  gqe_bench::cupti::user_range_profiler profiler(config);

  double first_profile = 0.0;

  for (int32_t run = 0; run < runs; ++run) {
    profiler.start();
    run_cuda_fn();
    auto profile = profiler.stop();

    EXPECT_TRUE(profile.metric_values.contains(metric));
    EXPECT_GT(profile.metric_values[metric], 0);

    // SM instructions executed should be nearly identical across runs. On
    // L40S (original gqe test, 2025-10-07) the profiled values were exactly
    // identical over 5 runs; a small tolerance guards against variance on
    // other GPU models so the test isn't flaky.
    if (run == 0) {
      first_profile = profile.metric_values[metric];
    } else {
      EXPECT_NEAR(profile.metric_values[metric], first_profile, /* abs_error = */ 10.0);
    }
  }
}

/**
 * @brief Profile multiple metrics in a single run.
 */
TEST_F(UserRangeProfilerTest, MultiMetric)
{
  std::vector<std::string> const metrics = {"sm__inst_executed.sum", "smsp__warps_launched.sum"};

  gqe_bench::cupti::user_range_profiler::configuration config;
  config.device_id = gqe_bench::current_cuda_device();
  config.metrics   = metrics;

  gqe_bench::cupti::user_range_profiler profiler(config);
  profiler.start();
  run_cuda_fn();
  auto profile = profiler.stop();

  for (auto const& m : metrics) {
    EXPECT_TRUE(profile.metric_values.contains(m));
    EXPECT_GT(profile.metric_values[m], 0);
  }
}

/**
 * @brief Recover from a thrown exception.
 */
TEST_F(UserRangeProfilerTest, ExceptionRecovery)
{
  bool exception_was_caught = false;

  try {
    gqe_bench::cupti::user_range_profiler::configuration config;
    config.device_id = gqe_bench::current_cuda_device();
    config.metrics   = {metric};

    gqe_bench::cupti::user_range_profiler profiler(config);
    profiler.start();
    throw std::runtime_error("An exception occurred.");
  } catch (std::runtime_error const&) {
    exception_was_caught = true;
  }
  EXPECT_TRUE(exception_was_caught);

  // New profiler instance must work correctly.
  gqe_bench::cupti::user_range_profiler::configuration config;
  config.device_id = gqe_bench::current_cuda_device();
  config.metrics   = {metric};

  gqe_bench::cupti::user_range_profiler profiler(config);
  profiler.start();
  run_cuda_fn();
  auto profile = profiler.stop();

  EXPECT_TRUE(profile.metric_values.contains(metric));
  EXPECT_GT(profile.metric_values[metric], 0);
}

/**
 * @brief The declared move constructor on `user_range_profiler` must produce a
 * usable profiler instance.
 */
TEST_F(UserRangeProfilerTest, MoveConstructor)
{
  gqe_bench::cupti::user_range_profiler::configuration config;
  config.device_id = gqe_bench::current_cuda_device();
  config.metrics   = {metric};

  gqe_bench::cupti::user_range_profiler original(config);
  gqe_bench::cupti::user_range_profiler moved(std::move(original));

  moved.start();
  run_cuda_fn();
  auto profile = moved.stop();

  EXPECT_TRUE(profile.metric_values.contains(metric));
  EXPECT_GT(profile.metric_values[metric], 0);
}

class ActivityProfilerTest : public CuptiTestBase {};

/**
 * @brief A kernel launched between start() and stop() produces at least one kernel record.
 */
TEST_F(ActivityProfilerTest, KernelCaptured)
{
  gqe_bench::cupti::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});

  profiler.start();
  run_cuda_fn();
  auto records = profiler.stop();

  EXPECT_FALSE(records.kernels.empty());
  for (auto const& k : records.kernels) {
    EXPECT_FALSE(k.name.empty());
    EXPECT_GT(k.end_ns, k.start_ns);
  }

  auto time_breakdown = gqe_bench::cupti::activity_profiler::get_time_breakdown(records);
  EXPECT_GT(time_breakdown.compute_kernel_s, 0);
  EXPECT_EQ(time_breakdown.io_kernel_s, 0);
  EXPECT_EQ(time_breakdown.memcpy_s, 0);
  EXPECT_EQ(time_breakdown.mem_decompress_s, 0);
  EXPECT_EQ(time_breakdown.merged_io_activity_s, 0);
}

/**
 * @brief Records only contain events from within the start()/stop() window.
 */
TEST_F(ActivityProfilerTest, WindowIsolation)
{
  gqe_bench::cupti::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});

  run_cuda_fn();  // before start() — flushed and discarded by start()

  profiler.start();
  auto records = profiler.stop();

  EXPECT_TRUE(records.kernels.empty());
  EXPECT_TRUE(records.memcopies.empty());
  EXPECT_TRUE(records.markers.empty());
  EXPECT_TRUE(records.mem_decompress.empty());
}

/**
 * @brief The profiler can be reused across multiple start()/stop() cycles.
 */
TEST_F(ActivityProfilerTest, MultiUse)
{
  constexpr int32_t runs = 3;

  gqe_bench::cupti::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});

  for (int32_t run = 0; run < runs; ++run) {
    profiler.start();
    run_cuda_fn();
    auto records = profiler.stop();

    EXPECT_FALSE(records.kernels.empty()) << "Failed on run " << run;
  }
}

/**
 * @brief An NVTX scoped range pushed inside the window appears as a marker_event.
 */
TEST_F(ActivityProfilerTest, NvtxMarkerCaptured)
{
  constexpr auto range_name = "test_nvtx_range";

  gqe_bench::cupti::activity_profiler profiler(
    {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MARKER});

  profiler.start();
  {
    nvtx_scoped_range range(range_name);
    run_cuda_fn();
  }  // nvtxRangePop fires here
  auto records = profiler.stop();

  auto it = std::find_if(
    records.markers.begin(), records.markers.end(), [](gqe_bench::cupti::marker_event const& e) {
      return e.name.find(range_name) != std::string::npos;
    });

  EXPECT_NE(it, records.markers.end()) << "Expected marker_event named '" << range_name << "'";
  if (it != records.markers.end()) { EXPECT_GE(it->end_ns, it->start_ns); }
}

/**
 * @brief A host-to-device memcpy inside the window produces at least one memcpy record.
 */
TEST_F(ActivityProfilerTest, MemcpyCaptured)
{
  gqe_bench::cupti::activity_profiler profiler({CUPTI_ACTIVITY_KIND_MEMCPY});

  profiler.start();
  run_memcpy();
  auto records = profiler.stop();

  EXPECT_FALSE(records.memcopies.empty());
  for (auto const& m : records.memcopies) {
    EXPECT_GT(m.end_ns, m.start_ns);
    EXPECT_EQ(m.bytes, memcpy_num_bytes);
  }

  auto time_breakdown = gqe_bench::cupti::activity_profiler::get_time_breakdown(records);
  EXPECT_EQ(time_breakdown.compute_kernel_s, 0);
  EXPECT_EQ(time_breakdown.io_kernel_s, 0);
  EXPECT_GT(time_breakdown.memcpy_s, 0);
  EXPECT_EQ(time_breakdown.mem_decompress_s, 0);
  EXPECT_GT(time_breakdown.merged_io_activity_s, 0);
}

/**
 * @brief An exception thrown inside a start()/stop() bracket does not corrupt
 * subsequent uses via a freshly-constructed profiler.
 */
TEST_F(ActivityProfilerTest, ExceptionRecovery)
{
  bool exception_was_caught = false;

  try {
    gqe_bench::cupti::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});
    profiler.start();
    throw std::runtime_error("Simulated exception.");
  } catch (std::runtime_error const&) {
    exception_was_caught = true;
  }
  EXPECT_TRUE(exception_was_caught);

  // A new profiler constructed after the failed one must work correctly.
  gqe_bench::cupti::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});
  profiler.start();
  run_cuda_fn();
  auto records = profiler.stop();

  EXPECT_FALSE(records.kernels.empty());
}

/**
 * @brief Constructing a second activity_profiler while one already exists throws,
 * because CUPTI Activity API uses process-global buffer callbacks.
 */
TEST_F(ActivityProfilerTest, SingleInstanceEnforced)
{
  gqe_bench::cupti::activity_profiler first({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});

  EXPECT_THROW(
    { gqe_bench::cupti::activity_profiler second({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL}); },
    std::logic_error);
}

/**
 * @brief `get_time_breakdown` aggregates kernels whose names match the
 * caller-supplied filter_list into `io_kernel_s`. Default list only matches
 * GQE-specific kernel names; pass a custom list so the test workload's
 * kernel (`mac_kernel`) counts as an IO kernel.
 */
TEST_F(ActivityProfilerTest, IoKernelFilterMatches)
{
  gqe_bench::cupti::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});
  profiler.start();
  run_cuda_fn();
  auto records = profiler.stop();

  auto time_breakdown = gqe_bench::cupti::activity_profiler::get_time_breakdown(
    records, /* io_kernel_filter_list = */ {"mac_kernel"});

  // With the filter matching our workload kernel, it is counted as IO, not compute.
  EXPECT_GT(time_breakdown.io_kernel_s, 0);
  EXPECT_EQ(time_breakdown.compute_kernel_s, 0);
  EXPECT_GT(time_breakdown.merged_io_activity_s, 0);
}

/**
 * @brief Constructed with an empty kinds vector, no activity is enabled and
 * `stop()` yields empty record vectors (no CUPTI callbacks were registered
 * for any kind).
 */
TEST_F(ActivityProfilerTest, EmptyKinds)
{
  gqe_bench::cupti::activity_profiler profiler(/* kinds = */ {});
  profiler.start();
  run_cuda_fn();
  auto records = profiler.stop();

  EXPECT_TRUE(records.kernels.empty());
  EXPECT_TRUE(records.memcopies.empty());
  EXPECT_TRUE(records.markers.empty());
  EXPECT_TRUE(records.mem_decompress.empty());
}

/**
 * @brief The destructor of a profiler that is still running must not throw
 * (it is `noexcept` — catch+log inside the impl). Exercises the path where
 * `~profiler()` is invoked between `start()` and `stop()`.
 */
TEST_F(ActivityProfilerTest, DestructorWhileRunning)
{
  {
    gqe_bench::cupti::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});
    profiler.start();
    run_cuda_fn();
    // intentionally no stop() — dtor fires here
  }

  // A new profiler must still construct and run correctly after the implicit
  // teardown of the first.
  gqe_bench::cupti::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});
  profiler.start();
  run_cuda_fn();
  auto records = profiler.stop();
  EXPECT_FALSE(records.kernels.empty());
}
