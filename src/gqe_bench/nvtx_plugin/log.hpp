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

#include <spdlog/details/log_msg.h>
#include <spdlog/logger.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <string>
#include <string_view>

#include <nvtx_plugin/env.hpp>

namespace gqe_bench {
namespace detail {

/**
 * @brief Pattern flag rendering a level under the name the engine's drain writes.
 *
 * The drain names its own levels rather than deferring to spdlog, so `warn` has
 * no counterpart among spdlog's: `%l` renders `warning` and `%L` renders a
 * single character. spdlog offers no way to rename a level.
 */
class level_name_flag final : public spdlog::custom_flag_formatter {
 public:
  void format(spdlog::details::log_msg const& msg,
              std::tm const&,
              spdlog::memory_buf_t& dest) override
  {
    auto const name = level_name(msg.level);
    dest.append(name.data(), name.data() + name.size());
  }

  [[nodiscard]] std::unique_ptr<spdlog::custom_flag_formatter> clone() const override
  {
    return std::make_unique<level_name_flag>();
  }

 private:
  static std::string_view level_name(spdlog::level::level_enum level)
  {
    switch (level) {
      case spdlog::level::trace: return "trace";
      case spdlog::level::debug: return "debug";
      case spdlog::level::info: return "info";
      case spdlog::level::warn: return "warn";
      case spdlog::level::err: return "error";
      case spdlog::level::critical: return "critical";
      default: return "unknown";
    }
  }
};

}  // namespace detail

/**
 * @brief Return the logger used by the plugin.
 *
 * Writes to stderr in the line format the node manager's log drain applies to
 * every engine record:
 *
 *     [YYYY-MM-DD HH:MM:SS.ffffff] [<source>] [<level>] [thread <tid>] <message>
 *
 * The drain formats the fields spdlog already carries on a record, so the
 * pattern below reproduces it from the same values. The logger name supplies
 * the source field, placing the plugin in the `gqe-bench:<module>` namespace
 * that `gqe_bench.logger` applies to the Python layer's records, alongside the
 * engine's `server` and `gpu<rank>`.
 *
 * The verbosity is read from the environment variable named by
 * `k_log_level_env`, which `gqe_bench.recording` populates from `--log-level`.
 *
 * @note The easiest way to log messages is to use the `GQE_BENCH_LOG_*` macros.
 */
inline spdlog::logger& logger()
{
  static std::unique_ptr<spdlog::logger> const instance = []() {
    // Stays out of spdlog's process-wide registry, which the plugin shares with
    // the engine it loads into: an unregistered logger cannot collide with the
    // engine's by name, and registry-wide settings cannot cross between them.
    auto plugin_logger = std::make_unique<spdlog::logger>(
      "gqe-bench:plugin", std::make_shared<spdlog::sinks::stderr_color_sink_mt>());

    auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<detail::level_name_flag>('*').set_pattern(
      "[%Y-%m-%d %H:%M:%S.%f] [%n] [%*] [thread %t] %v");
    plugin_logger->set_formatter(std::move(formatter));

    // The plugin runs inside processes that CUPTI and NVTX call into across C
    // frames, where an escaping exception is undefined behaviour rather than a
    // reportable error. Divert sink failures to stderr instead of throwing.
    plugin_logger->set_error_handler(
      [](std::string const& msg) { std::fprintf(stderr, "[gqe-bench:plugin] %s\n", msg.c_str()); });

    auto const log_level = std::getenv(k_log_level_env);
    if (log_level && *log_level) { plugin_logger->set_level(spdlog::level::from_str(log_level)); }
    plugin_logger->flush_on(spdlog::level::err);

    return plugin_logger;
  }();

  return *instance;
}

}  // namespace gqe_bench

#define GQE_BENCH_LOG_DEBUG(...) gqe_bench::logger().debug(__VA_ARGS__)
#define GQE_BENCH_LOG_INFO(...)  gqe_bench::logger().info(__VA_ARGS__)
#define GQE_BENCH_LOG_WARN(...)  gqe_bench::logger().warn(__VA_ARGS__)
#define GQE_BENCH_LOG_ERROR(...) gqe_bench::logger().error(__VA_ARGS__)
