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

#include "run_writer.hpp"

#include "log.hpp"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>

namespace gqe_bench {

namespace {

/**
 * @brief Check a SQLite result code and throw `std::runtime_error` on any
 * code other than `SQLITE_OK`, `SQLITE_DONE`, or `SQLITE_ROW`.
 */
void sqlite_check(sqlite3* db, int rc, char const* op)
{
  if (rc == SQLITE_OK || rc == SQLITE_DONE || rc == SQLITE_ROW) return;
  throw std::runtime_error(std::format("{}: {}", op, sqlite3_errmsg(db)));
}

/**
 * @brief RAII wrapper for `BEGIN IMMEDIATE` ... `COMMIT` / `ROLLBACK`.
 *
 * `IMMEDIATE` acquires the reserved write lock at construction so the
 * busy timeout applies to a single well-defined wait. The destructor
 * rolls back unless `commit` already succeeded; it does not throw, so
 * stack unwinding through this object is safe.
 */
class write_transaction {
 public:
  explicit write_transaction(sqlite3* db) : _db(db)
  {
    int rc = sqlite3_exec(_db, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
      GQE_BENCH_LOG_ERROR("BEGIN IMMEDIATE failed: {}", sqlite3_errmsg(_db));
      _db = nullptr;
    }
  }

  ~write_transaction()
  {
    if (_db) sqlite3_exec(_db, "ROLLBACK", nullptr, nullptr, nullptr);
  }

  write_transaction(write_transaction const&)            = delete;
  write_transaction& operator=(write_transaction const&) = delete;

  [[nodiscard]] bool ok() const { return _db != nullptr; }

  [[nodiscard]] bool commit()
  {
    if (!_db) return false;
    int rc = sqlite3_exec(_db, "COMMIT", nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
      GQE_BENCH_LOG_ERROR("COMMIT failed: {}", sqlite3_errmsg(_db));
      return false;  // destructor rolls back while _db remains set
    }
    _db = nullptr;  // success: no rollback on destruction
    return true;
  }

 private:
  sqlite3* _db;
};

/**
 * @brief Helper that binds the activity-row prepared statements to one
 * rank's `(exp_id, run_num, gpu_info_id)` triple and steps a row per
 * event. The statements are owned by `run_writer`; this struct holds
 * non-owning copies for the duration of one rank's write.
 */
struct rank_row_writer {
  sqlite3* db                       = nullptr;
  sqlite3_stmt* kernel_stmt         = nullptr;
  sqlite3_stmt* memcpy_stmt         = nullptr;
  sqlite3_stmt* marker_stmt         = nullptr;
  sqlite3_stmt* mem_decompress_stmt = nullptr;
  std::int64_t exp_id               = 0;
  std::int64_t run_num              = 0;
  std::int64_t gpu_info_id          = 0;

  /**
   * @brief Reset `stmt` and bind the shared `(exp_id, run_num, gpu_info_id)`
   * triple into the next three columns, advancing `arg`.
   */
  void prepare_row(sqlite3_stmt* stmt, int& arg)
  {
    sqlite3_reset(stmt);
    sqlite3_bind_int64(stmt, arg++, exp_id);
    sqlite3_bind_int64(stmt, arg++, run_num);
    sqlite3_bind_int64(stmt, arg++, gpu_info_id);
  }

  void on_kernel(std::int64_t s, std::int64_t e, char const* name, std::size_t name_len)
  {
    int arg = 1;
    prepare_row(kernel_stmt, arg);
    sqlite3_bind_text(kernel_stmt, arg++, name, static_cast<int>(name_len), SQLITE_TRANSIENT);
    sqlite3_bind_int64(kernel_stmt, arg++, s);
    sqlite3_bind_int64(kernel_stmt, arg++, e);
    sqlite_check(db, sqlite3_step(kernel_stmt), "insert_kernel_activity step");
  }

  void on_memcpy(std::int64_t s, std::int64_t e, std::uint8_t kind, std::uint64_t bytes)
  {
    int arg = 1;
    prepare_row(memcpy_stmt, arg);
    sqlite3_bind_int64(memcpy_stmt, arg++, kind);
    sqlite3_bind_int64(memcpy_stmt, arg++, static_cast<std::int64_t>(bytes));
    sqlite3_bind_int64(memcpy_stmt, arg++, s);
    sqlite3_bind_int64(memcpy_stmt, arg++, e);
    sqlite_check(db, sqlite3_step(memcpy_stmt), "insert_memcpy_activity step");
  }

  void on_marker(std::int64_t s, std::int64_t e, char const* name, std::size_t name_len)
  {
    int arg = 1;
    prepare_row(marker_stmt, arg);
    sqlite3_bind_text(marker_stmt, arg++, name, static_cast<int>(name_len), SQLITE_TRANSIENT);
    sqlite3_bind_int64(marker_stmt, arg++, s);
    sqlite3_bind_int64(marker_stmt, arg++, e);
    sqlite_check(db, sqlite3_step(marker_stmt), "insert_marker_activity step");
  }

  void on_mem_decompress(std::int64_t s, std::int64_t e, std::uint64_t source_bytes)
  {
    int arg = 1;
    prepare_row(mem_decompress_stmt, arg);
    sqlite3_bind_int64(mem_decompress_stmt, arg++, static_cast<std::int64_t>(source_bytes));
    sqlite3_bind_int64(mem_decompress_stmt, arg++, s);
    sqlite3_bind_int64(mem_decompress_stmt, arg++, e);
    sqlite_check(db, sqlite3_step(mem_decompress_stmt), "insert_mem_decompress_activity step");
  }
};
}  // namespace

/**
 * @brief Prepared-statement wrapper for the `gqe_metric_info` dimension
 * table. Resolves metric names to their `m_id` keys. Defined here, not in
 * the header, to keep the type out of the public surface.
 */
class run_writer::metric_info_repo {
 public:
  explicit metric_info_repo(sqlite3* db) : _db(db)
  {
    constexpr char const* sql_insert = "INSERT OR IGNORE INTO gqe_metric_info (m_name) VALUES (?1)";
    constexpr char const* sql_select = "SELECT m_id FROM gqe_metric_info WHERE m_name = ?1";
    sqlite_check(_db,
                 sqlite3_prepare_v2(_db, sql_insert, -1, &_insert_stmt, nullptr),
                 "prepare metric_info insert");
    sqlite_check(_db,
                 sqlite3_prepare_v2(_db, sql_select, -1, &_select_stmt, nullptr),
                 "prepare metric_info select");
  }

  ~metric_info_repo()
  {
    sqlite3_finalize(_insert_stmt);
    sqlite3_finalize(_select_stmt);
  }

  metric_info_repo(metric_info_repo const&)            = delete;
  metric_info_repo& operator=(metric_info_repo const&) = delete;

  std::optional<std::int64_t> get_or_insert(char const* name)
  {
    sqlite3_reset(_insert_stmt);
    sqlite3_bind_text(_insert_stmt, 1, name, -1, SQLITE_STATIC);
    if (int rc = sqlite3_step(_insert_stmt); rc != SQLITE_DONE) {
      GQE_BENCH_LOG_ERROR("metric_info insert '{}' failed: {}", name, sqlite3_errmsg(_db));
      return std::nullopt;
    }

    sqlite3_reset(_select_stmt);
    sqlite3_bind_text(_select_stmt, 1, name, -1, SQLITE_STATIC);
    std::optional<std::int64_t> id;
    if (sqlite3_step(_select_stmt) == SQLITE_ROW) {
      id = sqlite3_column_int64(_select_stmt, 0);
    } else {
      GQE_BENCH_LOG_ERROR("metric_info_repo: lookup of '{}' returned no row", name);
    }
    // A SELECT stopped at SQLITE_ROW leaves an open cursor that keeps its transaction
    // active and the DB locked until reset (https://www.sqlite.org/lang_transaction.html).
    sqlite3_reset(_select_stmt);
    return id;
  }

 private:
  sqlite3* _db;
  sqlite3_stmt* _insert_stmt = nullptr;
  sqlite3_stmt* _select_stmt = nullptr;
};

run_writer::run_writer(std::string db_path, std::uint32_t total_ranks)
  : _db_path(std::move(db_path)), _total_ranks(total_ranks)
{
  sqlite3* raw = nullptr;
  int const rc = sqlite3_open_v2(_db_path.c_str(), &raw, SQLITE_OPEN_READWRITE, nullptr);
  if (rc != SQLITE_OK) {
    std::string msg{"run_writer: sqlite3_open_v2 failed for "};
    msg += _db_path;
    msg += ": ";
    msg += raw ? sqlite3_errmsg(raw) : sqlite3_errstr(rc);
    if (raw) sqlite3_close(raw);
    throw std::runtime_error(msg);
  }
  _db.reset(raw);
  sqlite3_busy_timeout(_db.get(), 5000);
  sqlite3_exec(_db.get(), "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr);
  GQE_BENCH_LOG_INFO("Opened DB: {}", _db_path);

  _metric_info = std::make_unique<metric_info_repo>(_db.get());
  prepare_statements();
  resolve_gpu_info_ids();
}

run_writer::~run_writer()
{
  // sqlite3_finalize is a no-op on nullptr.
  for (sqlite3_stmt* s : {_next_run_number_stmt,
                          _insert_run_stmt,
                          _insert_run_ext_stmt,
                          _insert_breakdown_stmt,
                          _select_gpu_info_id_stmt,
                          _insert_kernel_activity_stmt,
                          _insert_memcpy_activity_stmt,
                          _insert_marker_activity_stmt,
                          _insert_mem_decompress_stmt}) {
    sqlite3_finalize(s);
  }
}

void run_writer::prepare_statements()
{
  sqlite3* const db = _db.get();
  auto prep         = [&](char const* sql, sqlite3_stmt** out, char const* op) {
    sqlite_check(db, sqlite3_prepare_v2(db, sql, -1, out, nullptr), op);
  };

  constexpr char const* sql_next_run_number =
    "SELECT COALESCE(MAX(n), -1) + 1 FROM ("
    "  SELECT r_number AS n FROM run WHERE r_experiment_id = ?1"
    "  UNION ALL"
    "  SELECT fr_number AS n FROM failed_run WHERE fr_experiment_id = ?1"
    ")";
  constexpr char const* sql_insert_run =
    "INSERT INTO run (r_experiment_id, r_number, r_nvtx_marker, r_duration_s) "
    "VALUES (?1, ?2, ?3, ?4)";
  constexpr char const* sql_insert_run_ext =
    "INSERT INTO gqe_run_ext "
    "(re_experiment_id, re_run_number, re_metric_info_id, re_metric_value, re_gpu_info_id) "
    "VALUES (?1, ?2, ?3, ?4, ?5)";
  constexpr char const* sql_insert_breakdown =
    "INSERT INTO gqe_run_time_breakdown "
    "(tb_experiment_id, tb_run_number, tb_gpu_info_id, "
    " tb_in_memory_read_task_s, tb_compute_kernel_s, tb_io_kernel_s, "
    " tb_memcpy_s, tb_mem_decompress_s, tb_merged_io_activity_s) "
    "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)";
  constexpr char const* sql_select_gpu_info_id = "SELECT g_id FROM gpu_info WHERE g_gpu_uuid = ?1";
  constexpr char const* sql_insert_ka =
    "INSERT INTO gqe_run_cupti_kernel_activity "
    "(ka_experiment_id, ka_run_number, ka_gpu_info_id, ka_name, ka_start_time, ka_end_time) "
    "VALUES (?1, ?2, ?3, ?4, ?5, ?6)";
  constexpr char const* sql_insert_mca =
    "INSERT INTO gqe_run_cupti_memcpy_activity "
    "(mca_experiment_id, mca_run_number, mca_gpu_info_id, mca_memcpy_kind, "
    " mca_bytes, mca_start_time, mca_end_time) "
    "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)";
  constexpr char const* sql_insert_mra =
    "INSERT INTO gqe_run_cupti_marker_activity "
    "(mra_experiment_id, mra_run_number, mra_gpu_info_id, mra_name, "
    " mra_start_time, mra_end_time) "
    "VALUES (?1, ?2, ?3, ?4, ?5, ?6)";
  constexpr char const* sql_insert_mda =
    "INSERT INTO gqe_run_cupti_mem_decompress_activity "
    "(mda_experiment_id, mda_run_number, mda_gpu_info_id, mda_source_bytes, "
    " mda_start_time, mda_end_time) "
    "VALUES (?1, ?2, ?3, ?4, ?5, ?6)";

  prep(sql_next_run_number, &_next_run_number_stmt, "prepare next_run_number");
  prep(sql_insert_run, &_insert_run_stmt, "prepare insert_run");
  prep(sql_insert_run_ext, &_insert_run_ext_stmt, "prepare insert_run_ext");
  prep(sql_insert_breakdown, &_insert_breakdown_stmt, "prepare insert_breakdown");
  prep(sql_insert_ka, &_insert_kernel_activity_stmt, "prepare insert_kernel_activity");
  prep(sql_insert_mca, &_insert_memcpy_activity_stmt, "prepare insert_memcpy_activity");
  prep(sql_insert_mra, &_insert_marker_activity_stmt, "prepare insert_marker_activity");
  prep(sql_insert_mda, &_insert_mem_decompress_stmt, "prepare insert_mem_decompress_activity");
  prep(sql_select_gpu_info_id, &_select_gpu_info_id_stmt, "prepare select_gpu_info_id");
}

void run_writer::resolve_gpu_info_ids()
{
  _gpu_info_ids.resize(_total_ranks);
  for (std::uint32_t r = 0; r < _total_ranks; ++r) {
    // Resolve the rank's CUDA index to a UUID via the driver API so the
    // lookup honors `CUDA_VISIBLE_DEVICES`.
    CUuuid cu_uuid{};
    CUresult const rc = cuDeviceGetUuid(&cu_uuid, static_cast<CUdevice>(r));
    if (rc != CUDA_SUCCESS) {
      throw std::runtime_error(
        std::format("resolve_gpu_info_ids: cuDeviceGetUuid(cuda_index={}) failed: rc={}",
                    r,
                    static_cast<int>(rc)));
    }
    auto const* b = reinterpret_cast<unsigned char const*>(cu_uuid.bytes);
    char uuid_str[64];
    std::snprintf(uuid_str,
                  sizeof(uuid_str),
                  "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  b[0],
                  b[1],
                  b[2],
                  b[3],
                  b[4],
                  b[5],
                  b[6],
                  b[7],
                  b[8],
                  b[9],
                  b[10],
                  b[11],
                  b[12],
                  b[13],
                  b[14],
                  b[15]);
    sqlite3_reset(_select_gpu_info_id_stmt);
    sqlite3_bind_text(_select_gpu_info_id_stmt, 1, uuid_str, -1, SQLITE_TRANSIENT);
    if (sqlite3_step(_select_gpu_info_id_stmt) == SQLITE_ROW) {
      _gpu_info_ids[r] = sqlite3_column_int64(_select_gpu_info_id_stmt, 0);
    } else {
      throw std::runtime_error(std::format(
        "resolve_gpu_info_ids: gpu_info lookup miss for cuda_index={} uuid={}", r, uuid_str));
    }
    sqlite3_reset(_select_gpu_info_id_stmt);
  }
}

std::int64_t run_writer::next_run_number(std::int64_t exp_id)
{
  sqlite3_reset(_next_run_number_stmt);
  sqlite3_bind_int64(_next_run_number_stmt, 1, exp_id);
  int const rc = sqlite3_step(_next_run_number_stmt);
  if (rc != SQLITE_ROW) {
    sqlite3_reset(_next_run_number_stmt);
    sqlite_check(_db.get(), rc, "next_run_number step");
    throw std::runtime_error("next_run_number: unexpected non-ROW result");
  }
  std::int64_t const n = sqlite3_column_int64(_next_run_number_stmt, 0);
  sqlite3_reset(_next_run_number_stmt);
  return n;
}

void run_writer::write_run_row(std::int64_t exp_id, std::int64_t run_num, double duration_s)
{
  sqlite3_reset(_insert_run_stmt);
  sqlite3_bind_int64(_insert_run_stmt, 1, exp_id);
  sqlite3_bind_int64(_insert_run_stmt, 2, run_num);
  sqlite3_bind_null(_insert_run_stmt, 3);
  sqlite3_bind_double(_insert_run_stmt, 4, duration_s);
  sqlite_check(_db.get(), sqlite3_step(_insert_run_stmt), "insert_run step");
}

void run_writer::write_aggregated_stage_rows(std::int64_t exp_id,
                                             std::int64_t run_num,
                                             agg_protocol::aggregated_stages const& stages)
{
  // NULL `gpu_info_id` marks each row as the cross-rank reduction.
  std::array<double, ::gqe_bench::stages.size()> const stage_seconds{
    stages.build_s,
    stages.execute_s,
    stages.collect_s,
  };
  for (std::size_t i = 0; i < ::gqe_bench::stages.size(); ++i) {
    insert_metric_row(
      exp_id, run_num, std::nullopt, info_for(static_cast<stage>(i)).metric_name, stage_seconds[i]);
  }
}

void run_writer::write_rank_counters(std::int64_t exp_id,
                                     std::int64_t run_num,
                                     std::int64_t gpu_info_id,
                                     std::vector<std::pair<std::string, double>> const& counters)
{
  for (auto const& [name, value] : counters) {
    insert_metric_row(exp_id, run_num, gpu_info_id, name, value);
  }
}

void run_writer::write_rank_breakdown(std::int64_t exp_id,
                                      std::int64_t run_num,
                                      std::int64_t gpu_info_id,
                                      cupti::time_breakdown const& bd)
{
  sqlite3_reset(_insert_breakdown_stmt);
  sqlite3_bind_int64(_insert_breakdown_stmt, 1, exp_id);
  sqlite3_bind_int64(_insert_breakdown_stmt, 2, run_num);
  sqlite3_bind_int64(_insert_breakdown_stmt, 3, gpu_info_id);
  sqlite3_bind_double(_insert_breakdown_stmt, 4, bd.in_memory_read_task_s);
  sqlite3_bind_double(_insert_breakdown_stmt, 5, bd.compute_kernel_s);
  sqlite3_bind_double(_insert_breakdown_stmt, 6, bd.io_kernel_s);
  sqlite3_bind_double(_insert_breakdown_stmt, 7, bd.memcpy_s);
  sqlite3_bind_double(_insert_breakdown_stmt, 8, bd.mem_decompress_s);
  sqlite3_bind_double(_insert_breakdown_stmt, 9, bd.merged_io_activity_s);
  sqlite_check(_db.get(), sqlite3_step(_insert_breakdown_stmt), "insert_breakdown step");
}

void run_writer::write_rank_events(std::int64_t exp_id,
                                   std::int64_t run_num,
                                   std::int64_t gpu_info_id,
                                   cupti::activity_records const& events)
{
  rank_row_writer w{
    .db                  = _db.get(),
    .kernel_stmt         = _insert_kernel_activity_stmt,
    .memcpy_stmt         = _insert_memcpy_activity_stmt,
    .marker_stmt         = _insert_marker_activity_stmt,
    .mem_decompress_stmt = _insert_mem_decompress_stmt,
    .exp_id              = exp_id,
    .run_num             = run_num,
    .gpu_info_id         = gpu_info_id,
  };
  for (auto const& k : events.kernels) {
    w.on_kernel(k.start_ns, k.end_ns, k.name.data(), k.name.size());
  }
  for (auto const& m : events.memcopies) {
    w.on_memcpy(m.start_ns, m.end_ns, m.kind, m.bytes);
  }
  for (auto const& mk : events.markers) {
    w.on_marker(mk.start_ns, mk.end_ns, mk.name.data(), mk.name.size());
  }
  for (auto const& d : events.mem_decompress) {
    w.on_mem_decompress(d.start_ns, d.end_ns, d.source_bytes);
  }
}

void run_writer::insert_metric_row(std::int64_t exp_id,
                                   std::int64_t run_num,
                                   std::optional<std::int64_t> gpu_info_id,
                                   std::string_view metric_name,
                                   double value)
{
  // metric_name is a SQL bind source; ensure null-termination by copying.
  std::string name_copy{metric_name};
  std::optional<std::int64_t> m_id = _metric_info->get_or_insert(name_copy.c_str());
  if (!m_id) throw std::runtime_error("metric_info lookup failed for '" + name_copy + "'");
  sqlite3_reset(_insert_run_ext_stmt);
  sqlite3_bind_int64(_insert_run_ext_stmt, 1, exp_id);
  sqlite3_bind_int64(_insert_run_ext_stmt, 2, run_num);
  sqlite3_bind_int64(_insert_run_ext_stmt, 3, *m_id);
  sqlite3_bind_double(_insert_run_ext_stmt, 4, value);
  if (gpu_info_id) {
    sqlite3_bind_int64(_insert_run_ext_stmt, 5, *gpu_info_id);
  } else {
    sqlite3_bind_null(_insert_run_ext_stmt, 5);
  }
  sqlite_check(_db.get(), sqlite3_step(_insert_run_ext_stmt), "insert_run_ext step");
}

void run_writer::write_run(agg_protocol::gathered_run const& g, std::int64_t experiment_id)
{
  write_transaction txn{_db.get()};
  if (!txn.ok()) throw std::runtime_error("write_run: BEGIN IMMEDIATE failed");

  std::int64_t const run_num = next_run_number(experiment_id);

  write_run_row(experiment_id, run_num, g.stages.total_s);
  write_aggregated_stage_rows(experiment_id, run_num, g.stages);

  // Per-rank rows: counters + (optional) breakdown + (optional) events.
  // Order matches g.per_rank, with rank 0 first.
  for (std::uint32_t r = 0; r < g.per_rank.size(); ++r) {
    std::int64_t const gpu_info_id = _gpu_info_ids[r];
    auto const& d                  = g.per_rank[r];
    write_rank_counters(experiment_id, run_num, gpu_info_id, d.counters);
    if (d.breakdown_valid) {
      write_rank_breakdown(experiment_id, run_num, gpu_info_id, d.breakdown);
    }
    write_rank_events(experiment_id, run_num, gpu_info_id, d.events);
  }

  if (!txn.commit()) throw std::runtime_error("write_run: COMMIT failed");
  GQE_BENCH_LOG_INFO("Run recorded: e_id={} r_number={} r_duration_s={:.4f}s ranks={}",
                     experiment_id,
                     run_num,
                     g.stages.total_s,
                     _total_ranks);
}

}  // namespace gqe_bench
