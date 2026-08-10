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
#include <nvtx_plugin/stages.hpp>

#include <sqlite3.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gqe_bench {

/**
 * @brief Owner of the SQLite connection, prepared statements, and the
 * resolved per-rank `gpu_info_id` lookups. Used only on rank 0.
 *
 * `write_run` commits one run's worth of rows in a single
 * `BEGIN IMMEDIATE` / `COMMIT` transaction: one `run` row, one
 * aggregated row per stage, and the per-rank counter / breakdown / activity
 * rows for every rank. The per-run input is provided as a plain value
 * type, so this class has no dependency on shared memory or NVTX.
 *
 * Construction either succeeds fully (DB opened, statements prepared,
 * `gpu_info_id`s resolved) or throws. There is no half-constructed state.
 */
class run_writer {
 public:
  /**
   * @brief Open `db_path`, prepare statements, and resolve each rank's
   * `gpu_info_id`. Throws on any failure (DB open, statement prepare,
   * GPU resolution).
   *
   * @param[in] db_path     Path to the SQLite database opened by the
   *                        Python side.
   * @param[in] total_ranks Number of ranks whose data this writer will
   *                        receive.
   *
   * @throw std::runtime_error On `sqlite3_open_v2` failure or any
   *        statement-prepare failure.
   */
  run_writer(std::string db_path, std::uint32_t total_ranks);

  /**
   * @brief Finalize all prepared statements and close the database.
   */
  ~run_writer();

  run_writer(run_writer const&)            = delete;
  run_writer& operator=(run_writer const&) = delete;

  /**
   * @brief Commit one run's rows in a single transaction.
   *
   * Writes one `run` row with the cross-rank total duration, one
   * aggregated row per stage (with NULL `gpu_info_id` to mark the cross-rank
   * reduction), and the per-rank counter / breakdown / activity rows for
   * every rank. Rolls back on any SQLite error encountered partway
   * through.
   *
   * @param[in] g             Per-run data gathered from all ranks.
   * @param[in] experiment_id Foreign-key value for the `run` row.
   */
  void write_run(agg_protocol::gathered_run const& g, std::int64_t experiment_id);

 private:
  /**
   * @brief Prepared-statement wrapper for the `gqe_metric_info` dimension
   * table. Resolves metric names to their `m_id` keys. Definition lives
   * in `run_writer.cpp`; held as `unique_ptr` here to keep the class
   * file-private.
   */
  class metric_info_repo;

  /**
   * @brief Prepare every SQL statement used by this class.
   * @throw std::runtime_error On any prepare failure.
   */
  void prepare_statements();

  /**
   * @brief Resolve each CUDA rank index to its `gpu_info.g_id` via the
   * device UUID.
   *
   * @throw std::runtime_error If a rank's UUID cannot be read or is not
   *                           present in `gpu_info`.
   */
  void resolve_gpu_info_ids();

  /**
   * @brief Compute the next `r_number` for `exp_id` as one past the
   * maximum already used in `run` or `failed_run`.
   *
   * @throw std::runtime_error On SQLite failure or unexpected non-row
   *                           result.
   */
  std::int64_t next_run_number(std::int64_t exp_id);

  /**
   * @brief Insert one row into `run` with NULL `r_nvtx_marker`.
   */
  void write_run_row(std::int64_t exp_id, std::int64_t run_num, double duration_s);

  /**
   * @brief Insert one `gqe_run_ext` row per stage using the
   * cross-rank reduced durations. NULL `gpu_info_id` marks them as the
   * reduced view.
   */
  void write_aggregated_stage_rows(std::int64_t exp_id,
                                   std::int64_t run_num,
                                   agg_protocol::aggregated_stages const& stages);

  /**
   * @brief Insert one `gqe_run_ext` row per counter for the given rank.
   */
  void write_rank_counters(std::int64_t exp_id,
                           std::int64_t run_num,
                           std::int64_t gpu_info_id,
                           std::vector<std::pair<std::string, double>> const& counters);

  /**
   * @brief Insert one `gqe_run_time_breakdown` row for the given rank.
   */
  void write_rank_breakdown(std::int64_t exp_id,
                            std::int64_t run_num,
                            std::int64_t gpu_info_id,
                            cupti::time_breakdown const& bd);

  /**
   * @brief Insert per-event rows into the `gqe_run_cupti_*_activity`
   * tables for the given rank.
   */
  void write_rank_events(std::int64_t exp_id,
                         std::int64_t run_num,
                         std::int64_t gpu_info_id,
                         cupti::activity_records const& events);

  /**
   * @brief Insert one `gqe_run_ext` row for `metric_name` with the given
   * value; resolves the metric's `m_id` via `metric_info_repo`. A
   * `std::nullopt` `gpu_info_id` writes NULL for the GPU column.
   *
   * @throw std::runtime_error If the metric-info lookup fails.
   */
  void insert_metric_row(std::int64_t exp_id,
                         std::int64_t run_num,
                         std::optional<std::int64_t> gpu_info_id,
                         std::string_view metric_name,
                         double value);

  /**
   * @brief Database path retained for diagnostics.
   */
  std::string _db_path;

  /**
   * @brief Number of ranks expected in `gathered_run::per_rank`.
   */
  std::uint32_t _total_ranks;

  /**
   * @brief Owned SQLite connection; closed by the unique_ptr deleter.
   */
  std::unique_ptr<sqlite3, decltype(&sqlite3_close)> _db{nullptr, &sqlite3_close};

  /**
   * @brief Metric-name to `m_id` resolver.
   */
  std::unique_ptr<metric_info_repo> _metric_info;

  /**
   * @brief Per-rank `gpu_info.g_id`, resolved at construction.
   */
  std::vector<std::int64_t> _gpu_info_ids;

  /**
   * @brief Statement returning the next `r_number` for an experiment.
   */
  sqlite3_stmt* _next_run_number_stmt = nullptr;

  /**
   * @brief `INSERT INTO run` statement.
   */
  sqlite3_stmt* _insert_run_stmt = nullptr;

  /**
   * @brief `INSERT INTO gqe_run_ext` statement.
   */
  sqlite3_stmt* _insert_run_ext_stmt = nullptr;

  /**
   * @brief `INSERT INTO gqe_run_time_breakdown` statement.
   */
  sqlite3_stmt* _insert_breakdown_stmt = nullptr;

  /**
   * @brief `SELECT g_id FROM gpu_info WHERE g_gpu_uuid = ?` statement.
   */
  sqlite3_stmt* _select_gpu_info_id_stmt = nullptr;

  /**
   * @brief `INSERT INTO gqe_run_cupti_kernel_activity` statement.
   */
  sqlite3_stmt* _insert_kernel_activity_stmt = nullptr;

  /**
   * @brief `INSERT INTO gqe_run_cupti_memcpy_activity` statement.
   */
  sqlite3_stmt* _insert_memcpy_activity_stmt = nullptr;

  /**
   * @brief `INSERT INTO gqe_run_cupti_marker_activity` statement.
   */
  sqlite3_stmt* _insert_marker_activity_stmt = nullptr;

  /**
   * @brief `INSERT INTO gqe_run_cupti_mem_decompress_activity` statement.
   */
  sqlite3_stmt* _insert_mem_decompress_stmt = nullptr;
};

}  // namespace gqe_bench
