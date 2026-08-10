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

/**
 * @file plugin.cpp
 * @brief Plugin entry point and CUPTI subscription wiring.
 *
 * `libgqe_bench.so` is intended to be loaded via `LD_PRELOAD` and
 * subscribes to the CUPTI NVTX callback domain. It observes the outer
 * `execute_plan` range and the pipeline stages `build_task_graph`,
 * `execute_task_graph`, and `collect_results`. The names are a contract
 * with the engine that emits the NVTX markers.
 */

#include "plugin.hpp"

#include "log.hpp"
#include "nvtx_injection.hpp"
#include <nvtx_plugin/env.hpp>
#include <nvtx_plugin/stages.hpp>

#include <cupti_callbacks.h>
#include <cupti_nvtx_cbid.h>
#include <errno.h>  // program_invocation_short_name
#include <generated_nvtx_meta.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

namespace gqe_bench {

std::atomic<std::shared_ptr<plugin>> g_plugin;

namespace {

/**
 * @brief One entry per NVTX push, null for un-observed ranges, so
 * push/pop remains balanced under arbitrary interleaved NVTX traffic.
 */
thread_local std::vector<range_observer*> t_observer_stack;

/**
 * @brief Extract the ASCII range name from an NVTX `RangePushEx` callback
 * payload, or null if the message is not a plain ASCII string.
 */
char const* extract_push_name(CUpti_NvtxData const* data)
{
  auto const* params = static_cast<nvtxDomainRangePushEx_params const*>(data->functionParams);
  if (params->core.eventAttrib &&
      params->core.eventAttrib->messageType == NVTX_MESSAGE_TYPE_ASCII) {
    return params->core.eventAttrib->message.ascii;
  }
  return nullptr;
}

/**
 * @brief CUPTI subscription entry point.
 *
 * Dispatches `nvtxDomainRangePushEx` and `nvtxDomainRangePop` callbacks
 * to the matching observer, if any. The thread-local observer stack
 * carries one entry per push (possibly null) so push/pop stays balanced
 * even when NVTX ranges that no observer claims are interleaved.
 */
void CUPTIAPI cupti_callback(void* /*userdata*/,
                             CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid,
                             void const* cbdata)
{
  if (domain != CUPTI_CB_DOMAIN_NVTX) return;
  // `disable_plugin()` (called from another callback's exception path)
  // can store null into `g_plugin` concurrently. This local `shared_ptr`
  // keeps the plugin alive for the duration of this callback.
  auto p = g_plugin.load();
  if (!p) return;

  // CUPTI calls in via C ABI; an exception escaping this frame is UB.
  try {
    auto const* nvtx_data = static_cast<CUpti_NvtxData const*>(cbdata);

    if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangePushEx) {
      char const* name         = extract_push_name(nvtx_data);
      range_observer* observer = p->find_observer(name);
      t_observer_stack.push_back(observer);
      if (observer) observer->on_push(nvtx_data);
    } else if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangePop) {
      if (t_observer_stack.empty()) return;
      range_observer* observer = t_observer_stack.back();
      t_observer_stack.pop_back();
      if (observer) observer->on_pop();
    }
  } catch (std::exception const& e) {
    GQE_BENCH_LOG_ERROR("cupti_callback: caught exception: {}", e.what());
    disable_plugin();
  } catch (...) {
    GQE_BENCH_LOG_ERROR("cupti_callback: caught non-std exception");
    disable_plugin();
  }
}

/**
 * @brief Read shared-memory configuration values from the plugin env vars.
 *
 * @return The segment configuration, or `std::nullopt` if the segment-name
 *         env var is empty or unset, or any of the numeric env vars is
 *         missing or parses to zero. There is no representation of a
 *         segment that cannot be attached to.
 */
std::optional<shm_config> read_env_shm_config()
{
  char const* name_env = std::getenv(k_shm_name_env);
  if (!name_env || !*name_env) return std::nullopt;

  char const* num_ranks_env = std::getenv(k_num_ranks_env);
  char const* size_env      = std::getenv(k_shm_size_env);
  if (!num_ranks_env || !*num_ranks_env || !size_env || !*size_env) {
    GQE_BENCH_LOG_ERROR("multi-GPU: {} set but {} / {} missing; not attaching",
                        k_shm_name_env,
                        k_num_ranks_env,
                        k_shm_size_env);
    return std::nullopt;
  }

  char* end                           = nullptr;
  unsigned long const num_ranks       = std::strtoul(num_ranks_env, &end, 10);
  unsigned long long const size_bytes = std::strtoull(size_env, &end, 10);
  if (num_ranks == 0 || size_bytes == 0) {
    GQE_BENCH_LOG_ERROR("multi-GPU: {}={} and {}={} parse to zero; not attaching",
                        k_num_ranks_env,
                        num_ranks_env,
                        k_shm_size_env,
                        size_env);
    return std::nullopt;
  }

  shm_config cfg;
  cfg.name        = name_env;
  cfg.total_ranks = static_cast<std::uint32_t>(num_ranks);
  cfg.size        = static_cast<std::size_t>(size_bytes);
  GQE_BENCH_LOG_INFO(
    "multi-GPU shm config: name={} ranks={} size={}", name_env, num_ranks, size_bytes);
  return cfg;
}

/**
 * @brief Read the `time_breakdown` env var and log whether the feature is
 * enabled.
 */
bool read_env_time_breakdown()
{
  bool const enabled = env_flag(k_time_breakdown_env);
  GQE_BENCH_LOG_INFO("time_breakdown {}", enabled ? "ENABLED" : "disabled");
  return enabled;
}

/**
 * @brief Read the `cupti_metrics` env var and log whether the feature is
 * enabled.
 */
std::vector<std::string> read_env_cupti_metrics()
{
  auto m = env_str_list(k_cupti_metrics_env);
  if (!m.empty()) {
    GQE_BENCH_LOG_INFO("cupti_metrics ENABLED ({} metric(s))", m.size());
  } else {
    GQE_BENCH_LOG_INFO("cupti_metrics disabled");
  }
  return m;
}

/**
 * @brief Resolve every environment variable the plugin acts on.
 *
 * The single place the environment is consulted, so whether there is any
 * work to do is answerable here rather than inside observer construction.
 *
 * @return The resolved configuration, or `std::nullopt` if the database
 *         path or the shared-memory segment is absent. Both are required:
 *         every run is recorded through the segment, single- and multi-GPU
 *         alike, so without one there is nothing to observe.
 */
std::optional<plugin_config> read_env_config()
{
  char const* db_path = std::getenv(k_bench_db_env);
  if (!db_path || !*db_path) {
    GQE_BENCH_LOG_INFO("{} not set, profiling disabled pid={}", k_bench_db_env, getpid());
    return std::nullopt;
  }
  GQE_BENCH_LOG_INFO("Initializing (DB: {}) pid={}", db_path, getpid());

  std::optional<shm_config> shm = read_env_shm_config();
  if (!shm) {
    GQE_BENCH_LOG_WARN("{} not set, run recording disabled pid={}", k_shm_name_env, getpid());
    return std::nullopt;
  }

  plugin_config cfg;
  cfg.db_path        = db_path;
  cfg.shm            = *std::move(shm);
  cfg.time_breakdown = read_env_time_breakdown();
  cfg.cupti_metrics  = read_env_cupti_metrics();
  return cfg;
}

/**
 * @brief Build the observer registry.
 */
std::vector<std::unique_ptr<range_observer>> make_observers(plugin_config const& cfg)
{
  std::vector<std::unique_ptr<range_observer>> observers;
  observers.push_back(std::make_unique<execute_plan_observer>(cfg.db_path, cfg.shm));
  observers.push_back(std::make_unique<stage_observer>(stage::build));
  observers.push_back(
    std::make_unique<execute_stage_observer>(cfg.time_breakdown, cfg.cupti_metrics));
  observers.push_back(std::make_unique<stage_observer>(stage::collect));
  return observers;
}

}  // namespace

plugin::plugin(std::vector<std::unique_ptr<range_observer>> observers)
  : _observers(std::move(observers))
{
  // Observers must be populated before cuptiEnableDomain; cupti_callback
  // indexes `_observers` via find_observer() once the domain is enabled.
  GQE_BENCH_LOG_INFO("observers registered pid={} size={}", getpid(), _observers.size());

  CUptiResult sub_result = cuptiSubscribe(&_subscriber, cupti_callback, nullptr);
  if (sub_result != CUPTI_SUCCESS) {
    char const* msg = nullptr;
    cuptiGetResultString(sub_result, &msg);
    throw std::runtime_error(std::string{"cuptiSubscribe failed: "} + (msg ? msg : "unknown") +
                             " (code " + std::to_string(sub_result) + ")");
  }

  CUptiResult dom_result = cuptiEnableDomain(1, _subscriber, CUPTI_CB_DOMAIN_NVTX);
  if (dom_result != CUPTI_SUCCESS) {
    char const* msg = nullptr;
    cuptiGetResultString(dom_result, &msg);
    throw std::runtime_error(std::string{"cuptiEnableDomain failed: "} + (msg ? msg : "unknown") +
                             " (code " + std::to_string(dom_result) + ")");
  }

  GQE_BENCH_LOG_INFO("Initialization complete pid={}", getpid());
}

void disable_plugin() noexcept { g_plugin.store(nullptr); }

plugin::~plugin()
{
  if (_subscriber) cuptiUnsubscribe(_subscriber);
}

range_observer* plugin::find_observer(char const* name)
{
  if (!name) return nullptr;
  for (auto& o : _observers) {
    if (o && std::strcmp(o->name(), name) == 0) return o.get();
  }
  return nullptr;
}

namespace {

/**
 * @brief Library entry point invoked by the dynamic loader.
 *
 * Resolves the environment, configures NVTX injection, and (if the host
 * process is the expected one) builds the observers and constructs the
 * plugin.
 */
__attribute__((constructor)) void gqe_bench_init()
{
  std::setvbuf(stderr, nullptr, _IONBF, 0);

  // Returning before configure_nvtx_injection() leaves the host process without
  // the domain filter. That is deliberate: the filter only narrows what reaches
  // CUPTI, so installing it when nothing will be recorded costs other NVTX
  // consumers and gains nothing.
  std::optional<plugin_config> cfg = read_env_config();
  if (!cfg) return;

  configure_nvtx_injection();

  if (std::strcmp(program_invocation_short_name, k_task_manager_progname) != 0) {
    GQE_BENCH_LOG_INFO("progname '{}' != '{}', plugin observations disabled pid={}",
                       program_invocation_short_name,
                       k_task_manager_progname,
                       getpid());
    return;
  }

  try {
    auto p = std::make_shared<plugin>(make_observers(*cfg));
    g_plugin.store(std::move(p));
  } catch (std::exception const& e) {
    GQE_BENCH_LOG_ERROR("init failed: {}; profiling disabled pid={}", e.what(), getpid());
    // Revert the NVTX_INJECTION64_PATH we set earlier so the host process
    // isn't left with an orphan env var pointing at a plugin that never loaded.
    unconfigure_nvtx_injection();
  } catch (...) {
    GQE_BENCH_LOG_ERROR("init failed (non-std exception); profiling disabled pid={}", getpid());
    unconfigure_nvtx_injection();
  }
}

/**
 * @brief Release the plugin before static destruction begins.
 *
 * Runs ahead of every static destructor, so `~plugin()` — and the
 * `cuptiUnsubscribe` inside it — happens while `g_plugin` and the logger are
 * both still alive. Callbacks arriving afterwards load a null `g_plugin` and
 * return before touching either.
 */
__attribute__((destructor)) void gqe_bench_fini() { disable_plugin(); }

}  // namespace

}  // namespace gqe_bench
