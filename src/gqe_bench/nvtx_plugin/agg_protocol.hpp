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

/**
 * @file agg_protocol.hpp
 * @brief Shared-memory aggregation protocol for the multi-GPU plugin.
 *
 * Defines the per-rank slot layout, the segment header, and the
 * `agg_protocol` class that publishes one rank's per-run data into a
 * shared-memory segment and (on rank 0) gathers every rank's contribution.
 * Atomic accesses use `std::atomic_ref` over plain scalar storage so the
 * mmap'ed bytes need no in-place atomic construction.
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <utility>
#include <vector>

#include "cupti_wrappers/cupti_activity.hpp"
#include <nvtx_plugin/stages.hpp>

namespace gqe_bench {

/**
 * @brief Maximum CUPTI counter name length, including the trailing NUL.
 */
constexpr std::size_t k_max_counter_name_len = 48;

/**
 * @brief Maximum number of CUPTI counter values stored per slot. Counter
 * values are reduced per-rank by the range profiler's `stop()`, so the
 * cap is the count of metrics requested via the metrics env var.
 */
constexpr std::size_t k_max_counters = 16;

/**
 * @brief Per-slot payload capacity in bytes. Larger event sets are split
 * into multiple chunks by `agg_protocol::publish`.
 */
constexpr std::size_t k_payload_bytes = 4 * 1024 * 1024;

/**
 * @brief One CUPTI counter name + value pair.
 */
struct counter_entry {
  char name[k_max_counter_name_len];
  double value;
};
static_assert(sizeof(counter_entry) == 56, "counter_entry layout drift");

/**
 * @brief Header at the start of the shared-memory segment.
 */
struct agg_header {
  /**
   * @brief Attach-barrier counter; accessed via `atomic_ref<uint32_t>`.
   *
   * Every rank atomically increments it after attaching, then spins
   * until it reads `total_ranks`. Once past the barrier no rank touches
   * it again.
   */
  std::uint32_t ready_count;

  /**
   * @brief Per-run experiment_id slot; accessed via `atomic_ref<int32_t>`.
   *
   * `0` is the sentinel: an `execute_plan` push observed with this
   * value is skipped. Any other value is the experiment_id under which
   * rank 0 records the run.
   */
  std::int32_t requested_experiment_id;
};
static_assert(sizeof(agg_header) == 8, "agg_header layout drift");

/**
 * @brief Per-rank slot in the shared-memory segment.
 *
 * Each slot is a view into the mmap'ed segment, reached via
 * `agg_protocol::slot_at`. Members are ordered by alignment (8-byte first,
 * 4-byte next, 1-byte flags, then the byte-aligned `payload`) so the
 * compiler needs no internal padding. The 4 MiB `payload` makes any stack
 * or heap allocation a stack-overflow hazard on threads with a small
 * default pthread stack, so all special members are deleted.
 */
struct agg_slot {
  /**
   * @brief Publisher → consumer counter; accessed via `atomic_ref<uint64_t>`.
   *
   * Zero is the "never published" sentinel; the first publish writes 1.
   * The publisher increments `gen` before each chunk and rank 0 reads
   * each new value before the publisher overwrites the slot.
   */
  std::uint64_t gen;

  /**
   * @brief Consumer → publisher acknowledgement; accessed via
   * `atomic_ref<uint64_t>`.
   *
   * Rank 0 release-stores the gen it just consumed; the publisher spins
   * on `consumed_gen == own_gen` before overwriting the slot.
   */
  std::uint64_t consumed_gen;

  /**
   * @brief Per-stage `steady_clock` push timestamps in nanoseconds,
   * indexed by `stage` (build / execute / collect).
   */
  std::int64_t stage_starts_ns[stages.size()];
  /**
   * @brief Per-stage `steady_clock` pop timestamps in nanoseconds,
   * indexed by `stage` (build / execute / collect).
   */
  std::int64_t stage_ends_ns[stages.size()];

  /**
   * @brief In-memory read-task time breakdown component, in seconds.
   */
  double tb_in_memory_read_task_s;
  /**
   * @brief Compute-kernel time breakdown component, in seconds.
   */
  double tb_compute_kernel_s;
  /**
   * @brief I/O kernel time breakdown component, in seconds.
   */
  double tb_io_kernel_s;
  /**
   * @brief Memcpy time breakdown component, in seconds.
   */
  double tb_memcpy_s;
  /**
   * @brief Mem-decompress time breakdown component, in seconds.
   */
  double tb_mem_decompress_s;
  /**
   * @brief Merged-I/O activity time breakdown component, in seconds.
   */
  double tb_merged_io_activity_s;

  /**
   * @brief Per-rank reduced counter values.
   */
  counter_entry counters[k_max_counters];

  /**
   * @brief Range `[0, k_max_counters]`; only `counters[0..num_counters)`
   * are populated.
   */
  std::uint32_t num_counters;

  /**
   * @brief Kernel events in this chunk's payload.
   */
  std::uint32_t num_kernels;
  /**
   * @brief Memcpy events in this chunk's payload.
   */
  std::uint32_t num_memcopies;
  /**
   * @brief Marker events in this chunk's payload.
   */
  std::uint32_t num_markers;
  /**
   * @brief Mem-decompress events in this chunk's payload.
   */
  std::uint32_t num_mem_decompress;

  /**
   * @brief Valid bytes in `payload` for this chunk.
   */
  std::uint32_t payload_size;

  /**
   * @brief 1 on the rank's last chunk for the current run, 0 otherwise.
   */
  std::uint8_t last_chunk;

  /**
   * @brief 1 iff the `tb_*` fields above are populated.
   */
  std::uint8_t breakdown_valid;

  /**
   * @brief Serialized event records.
   *
   * Layout, in order:
   *   `[ kernel_event   x num_kernels ]`
   *   `[ memcpy_event   x num_memcopies ]`
   *   `[ marker_event   x num_markers ]`
   *   `[ mem_decompress x num_mem_decompress ]`
   * Names on kernel and marker events are length-prefixed; names longer
   * than 65535 bytes are truncated to that length.
   */
  std::uint8_t payload[k_payload_bytes];

  agg_slot()                           = delete;
  agg_slot(agg_slot const&)            = delete;
  agg_slot(agg_slot&&)                 = delete;
  agg_slot& operator=(agg_slot const&) = delete;
  agg_slot& operator=(agg_slot&&)      = delete;
};
static_assert(sizeof(agg_slot) == 4195344, "agg_slot layout drift");

static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free,
              "atomic_ref<uint32_t> must be lock-free for shm use");
static_assert(std::atomic_ref<std::int32_t>::is_always_lock_free,
              "atomic_ref<int32_t> must be lock-free for shm use");
static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free,
              "atomic_ref<uint64_t> must be lock-free for shm use");

/**
 * @brief Aggregation protocol over the shared-memory segment.
 *
 * One instance per process. Move-only: copying would share mapping
 * ownership, which the `munmap` in the destructor does not support.
 */
class agg_protocol {
 public:
  /**
   * @brief Per-stage durations after cross-rank reduction.
   *
   * Build and execute span the earliest start to the latest end across
   * all ranks; collect runs only on rank 0 and is reported unchanged.
   */
  struct aggregated_stages {
    /**
     * @brief Reduced build-stage duration in seconds.
     */
    double build_s;
    /**
     * @brief Reduced execute-stage duration in seconds.
     */
    double execute_s;
    /**
     * @brief Rank-0 collect-stage duration in seconds.
     */
    double collect_s;
    /**
     * @brief `build_s + execute_s`.
     */
    double total_s;
  };

  /**
   * @brief One rank's contribution to a run.
   *
   * Holds the per-stage timestamps used by the cross-rank reduction
   * together with the per-rank values written into the database without
   * further reduction (counters, time-breakdown, activity events).
   */
  struct per_rank_data {
    /**
     * @brief Per-stage push timestamps in ns, indexed by `stage`.
     */
    std::array<std::int64_t, stages.size()> stage_starts_ns{};
    /**
     * @brief Per-stage pop timestamps in ns, indexed by `stage`.
     */
    std::array<std::int64_t, stages.size()> stage_ends_ns{};
    /**
     * @brief Per-rank time breakdown.
     */
    cupti::time_breakdown breakdown{};
    /**
     * @brief True iff `breakdown` was populated.
     */
    bool breakdown_valid = false;
    /**
     * @brief Per-rank counter values.
     */
    std::vector<std::pair<std::string, double>> counters{};
    /**
     * @brief Per-rank activity events.
     */
    cupti::activity_records events{};
  };

  /**
   * @brief Result of `gather_run`: aggregated stage durations plus the
   * per-rank data for every rank in `[0, total_ranks)`.
   */
  struct gathered_run {
    /**
     * @brief Cross-rank reduced stage durations.
     */
    aggregated_stages stages;
    /**
     * @brief Per-rank data, in rank order.
     */
    std::vector<per_rank_data> per_rank;
  };

  /**
   * @brief Attach this process to an existing, pre-sized segment.
   *
   * @param[in] shm_name    POSIX segment name (with leading '/').
   * @param[in] env_size    Total segment size in bytes. Checked against
   *                        the local layout (`sizeof(agg_header) +
   *                        total_ranks * sizeof(agg_slot)`) as drift
   *                        detection.
   * @param[in] total_ranks Number of ranks participating in this segment.
   * @param[in] rank        This process's rank in `[0, total_ranks)`.
   *                        The caller's CUDA context must already be
   *                        bound to the rank's device.
   *
   * @throw std::runtime_error If the segment is missing, `mmap` fails,
   *                           the size disagrees with the layout, or
   *                           `rank >= total_ranks`.
   * @return A constructed `agg_protocol` that owns the mapping until
   *         destruction.
   */
  static agg_protocol attach(char const* shm_name,
                             std::size_t env_size,
                             std::uint32_t total_ranks,
                             std::uint32_t rank);

  /**
   * @brief Release the segment mapping. Does not unlink the segment; the
   * code that created it is responsible for `shm_unlink`. The mapping
   * itself is owned by `_base`, whose deleter calls `munmap`.
   */
  ~agg_protocol() = default;

  agg_protocol(agg_protocol const&)            = delete;
  agg_protocol& operator=(agg_protocol const&) = delete;

  agg_protocol(agg_protocol&&) noexcept            = default;
  agg_protocol& operator=(agg_protocol&&) noexcept = default;

  /**
   * @brief This process's rank.
   */
  [[nodiscard]] std::uint32_t rank() const noexcept { return _rank; }

  /**
   * @brief Number of ranks participating in this segment.
   */
  [[nodiscard]] std::uint32_t total_ranks() const noexcept { return _total_ranks; }

  /**
   * @brief True iff this process is rank 0.
   */
  [[nodiscard]] bool is_rank_zero() const noexcept { return _rank == 0; }

  /**
   * @brief Block until every rank has crossed the attach barrier.
   *
   * Atomically increments the header's `ready_count` once and spins on
   * it until it reads `total_ranks`. Called once per process,
   * immediately after `attach`.
   */
  void barrier() noexcept;

  /**
   * @brief Acquire-load the experiment_id slot from the header.
   *
   * Returns `0` (the sentinel) when no request is posted; any other
   * value is the experiment_id rank 0 records this run under.
   */
  [[nodiscard]] std::int32_t requested_experiment_id() const noexcept;

  /**
   * @brief Return a reference to rank `r`'s slot in the segment.
   *
   * The non-const overload is used to fill the caller's own slot before
   * publishing; the const overload is used to read a peer rank's slot
   * after that peer has finished publishing.
   */
  [[nodiscard]] agg_slot& slot_at(std::uint32_t r) noexcept;

  /**
   * @brief Return a const reference to rank `r`'s slot in the segment.
   */
  [[nodiscard]] agg_slot const& slot_at(std::uint32_t r) const noexcept;

  /**
   * @brief Reset per-run state. `_own_gen` and `_expected_gen[*]` go
   * back to 1 and `_self_events` is cleared. Called at the start of
   * every run.
   */
  void reset_run_state() noexcept;

  /**
   * @brief Publish this rank's per-run data into its slot.
   *
   * Fills the rank's own slot scalars from `ctx` and publishes its
   * activity events through the protocol. `ctx.activity_records` is
   * moved out; the caller's next `reset()` re-initializes it.
   */
  void publish_self(struct run_context& ctx);

  /**
   * @brief Build every rank's `per_rank_data` and compute the cross-rank
   * reduction. Build and execute use the earliest start to the latest
   * end across all ranks; collect uses rank 0's value unchanged. Rank 0
   * only.
   *
   * @return The aggregated stage durations and per-rank data, in rank
   *         order.
   */
  gathered_run gather_run();

 private:
  /**
   * @brief Custom deleter for the mmap'd segment held in `_base`. The
   * mapped size is carried alongside the pointer so the destructor can
   * call `munmap(ptr, size)` without a separate size member.
   */
  struct munmap_deleter {
    std::size_t size = 0;
    void operator()(void* p) const noexcept
    {
      if (p) ::munmap(p, size);
    }
  };

  /**
   * @brief Construct from an already-mapped segment. Used by `attach`.
   */
  agg_protocol(std::unique_ptr<void, munmap_deleter> base,
               std::uint32_t rank,
               std::uint32_t total_ranks) noexcept;

  /**
   * @brief Pointer to the segment's header.
   */
  agg_header* header() const noexcept { return static_cast<agg_header*>(_base.get()); }

  /**
   * @brief Pointer to the first slot. The slot array begins after the
   * header.
   */
  agg_slot* slots() const noexcept;

  /**
   * @brief Copy the per-stage timestamps, optional time breakdown, and
   * counter values from `ctx` into `slot`'s scalar fields.
   */
  static void fill_scalar_fields(agg_slot& slot, struct run_context const& ctx) noexcept;

  /**
   * @brief Publish this rank's activity events.
   *
   * For rank 0, moves `activity` into `_self_events`; the chunked path
   * is unused because rank 0 would have to acknowledge its own chunks.
   * For peers, serializes `activity` into the slot via the chunked
   * protocol.
   *
   * @param[in]     activity Events to serialize.
   * @param[in,out] own_gen  The publisher's chunk counter; advances by
   *                         the number of chunks emitted so it remains
   *                         monotonic across queries.
   */
  void publish(cupti::activity_records activity, std::uint64_t& own_gen);

  /**
   * @brief Build one rank's `per_rank_data`. For rank 0 (self), reads
   * the slot scalars and consumes `_self_events`. For peers, drains the
   * chunked event stream and reads the slot scalars after the
   * final-chunk acquire (the publisher writes the scalars before
   * advancing `gen`). Used by `gather_run`.
   *
   * @param[in] r Rank to drain.
   * @return The rank's `per_rank_data` with events, scalars, and
   *         counters populated.
   */
  per_rank_data drain_one_rank(std::uint32_t r);

  /**
   * @brief Owning handle to the mmap'd segment. The deleter holds the
   * mapped size and calls `munmap` on destruction. Null after move.
   */
  std::unique_ptr<void, munmap_deleter> _base;

  /**
   * @brief This process's rank.
   */
  std::uint32_t _rank = 0;

  /**
   * @brief Number of participating ranks.
   */
  std::uint32_t _total_ranks = 0;

  /**
   * @brief Rank-0-only buffer for own activity events. Set by `publish`
   * (when called from rank 0) and consumed by `drain_one_rank` (likewise);
   * rank 0 does not route its own events through the chunked protocol.
   */
  std::optional<cupti::activity_records> _self_events;

  /**
   * @brief Per-peer drain bookkeeping (rank 0 only). Sized to
   * `_total_ranks` at attach; reset to all-1 by `reset_run_state`.
   */
  std::vector<std::uint64_t> _expected_gen;

  /**
   * @brief Publisher's chunk counter (non-rank-0). Reset to 1 by
   * `reset_run_state`; advances per chunk emitted by `publish`.
   */
  std::uint64_t _own_gen = 1;
};

}  // namespace gqe_bench
