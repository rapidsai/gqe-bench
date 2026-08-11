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

#include "agg_protocol.hpp"

#include "log.hpp"
#include "run_context.hpp"
#include <nvtx_plugin/stages.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>

namespace gqe_bench {

namespace {

/**
 * @brief Render `strerror(errno)` as a `std::string` for error messages.
 */
std::string errno_string(int saved_errno) { return std::string{std::strerror(saved_errno)}; }

// Per-event byte format inside an `agg_slot::payload`:
//
//   kernel:         int64 start_ns, int64 end_ns, uint16 name_len, char[name_len]
//   memcpy:         int64 start_ns, int64 end_ns, uint8  kind, uint64 bytes
//   marker:         int64 start_ns, int64 end_ns, uint16 name_len, char[name_len]
//   mem_decompress: int64 start_ns, int64 end_ns, uint64 source_bytes
//
// Names longer than `k_max_name_len` bytes are truncated to that length.

/**
 * @brief Fixed-prefix size for kernel and marker events.
 */
constexpr std::size_t k_kernel_marker_header = 8 + 8 + 2;

/**
 * @brief Total size of one memcpy event.
 */
constexpr std::size_t k_memcpy_size = 8 + 8 + 1 + 8;

/**
 * @brief Total size of one mem-decompress event.
 */
constexpr std::size_t k_mem_decompress_size = 8 + 8 + 8;

/**
 * @brief Maximum encoded name length (uint16 cap).
 */
constexpr std::uint16_t k_max_name_len = 0xFFFF;

/**
 * @brief Read a little-endian `uint16` from `src` without imposing alignment.
 */
std::uint16_t read_u16(std::uint8_t const* src) noexcept
{
  std::uint16_t v;
  std::memcpy(&v, src, sizeof(v));
  return v;
}

/**
 * @brief Read a little-endian `int64` from `src` without imposing alignment.
 */
std::int64_t read_i64(std::uint8_t const* src) noexcept
{
  std::int64_t v;
  std::memcpy(&v, src, sizeof(v));
  return v;
}

/**
 * @brief Read a little-endian `uint64` from `src` without imposing alignment.
 */
std::uint64_t read_u64(std::uint8_t const* src) noexcept
{
  std::uint64_t v;
  std::memcpy(&v, src, sizeof(v));
  return v;
}

/**
 * @brief Return the on-wire byte size of one kernel or marker event whose
 * name is `name` (truncated to `k_max_name_len` bytes if longer).
 */
std::size_t kernel_or_marker_size(std::string const& name) noexcept
{
  std::size_t n = name.size();
  if (n > k_max_name_len) n = k_max_name_len;
  return k_kernel_marker_header + n;
}

/**
 * @brief Write a little-endian `uint16` to `dst` without imposing alignment.
 */
void write_u16(std::uint8_t* dst, std::uint16_t v) noexcept { std::memcpy(dst, &v, sizeof(v)); }

/**
 * @brief Write a little-endian `int64` to `dst` without imposing alignment.
 */
void write_i64(std::uint8_t* dst, std::int64_t v) noexcept { std::memcpy(dst, &v, sizeof(v)); }

/**
 * @brief Write a little-endian `uint64` to `dst` without imposing alignment.
 */
void write_u64(std::uint8_t* dst, std::uint64_t v) noexcept { std::memcpy(dst, &v, sizeof(v)); }

/**
 * @brief Encode one kernel or marker event into the byte buffer at `dst`,
 * truncating the name to `k_max_name_len` bytes if necessary.
 *
 * @return Number of bytes written.
 */
std::size_t encode_kernel_or_marker(std::uint8_t* dst,
                                    std::int64_t start_ns,
                                    std::int64_t end_ns,
                                    std::string const& name) noexcept
{
  std::size_t n = name.size();
  if (n > k_max_name_len) n = k_max_name_len;
  write_i64(dst + 0, start_ns);
  write_i64(dst + 8, end_ns);
  write_u16(dst + 16, static_cast<std::uint16_t>(n));
  std::memcpy(dst + 18, name.data(), n);
  return k_kernel_marker_header + n;
}

/**
 * @brief Encode one memcpy event into the byte buffer at `dst`.
 *
 * @return Number of bytes written.
 */
std::size_t encode_memcpy(std::uint8_t* dst,
                          std::int64_t start_ns,
                          std::int64_t end_ns,
                          std::uint8_t kind,
                          std::uint64_t bytes) noexcept
{
  write_i64(dst + 0, start_ns);
  write_i64(dst + 8, end_ns);
  dst[16] = kind;
  write_u64(dst + 17, bytes);
  return k_memcpy_size;
}

/**
 * @brief Encode one mem-decompress event into the byte buffer at `dst`.
 *
 * @return Number of bytes written.
 */
std::size_t encode_mem_decompress(std::uint8_t* dst,
                                  std::int64_t start_ns,
                                  std::int64_t end_ns,
                                  std::uint64_t source_bytes) noexcept
{
  write_i64(dst + 0, start_ns);
  write_i64(dst + 8, end_ns);
  write_u64(dst + 16, source_bytes);
  return k_mem_decompress_size;
}

/**
 * @brief On-wire size of one serialized event in shared memory.
 */
std::size_t size_in_shmem(cupti::kernel_event const& e) noexcept
{
  return kernel_or_marker_size(e.name);
}
std::size_t size_in_shmem(cupti::memcpy_event const&) noexcept { return k_memcpy_size; }
std::size_t size_in_shmem(cupti::marker_event const& e) noexcept
{
  return kernel_or_marker_size(e.name);
}
std::size_t size_in_shmem(cupti::mem_decompress_event const&) noexcept
{
  return k_mem_decompress_size;
}

/**
 * @brief Encode one event into the byte buffer at `dst`.
 *
 * @return Number of bytes written.
 */
std::size_t encode_into(std::uint8_t* dst, cupti::kernel_event const& e) noexcept
{
  return encode_kernel_or_marker(dst, e.start_ns, e.end_ns, e.name);
}
std::size_t encode_into(std::uint8_t* dst, cupti::memcpy_event const& e) noexcept
{
  return encode_memcpy(dst, e.start_ns, e.end_ns, e.kind, e.bytes);
}
std::size_t encode_into(std::uint8_t* dst, cupti::marker_event const& e) noexcept
{
  return encode_kernel_or_marker(dst, e.start_ns, e.end_ns, e.name);
}
std::size_t encode_into(std::uint8_t* dst, cupti::mem_decompress_event const& e) noexcept
{
  return encode_mem_decompress(dst, e.start_ns, e.end_ns, e.source_bytes);
}

/**
 * @brief Decode one kernel or marker event from `src` into `start_ns`,
 * `end_ns`, and `name`. The name field is filled with a copy of the
 * name bytes.
 *
 * @return Number of bytes consumed.
 */
std::size_t decode_kernel_or_marker(std::uint8_t const* src,
                                    std::int64_t& start_ns,
                                    std::int64_t& end_ns,
                                    std::string& name)
{
  start_ns                     = read_i64(src + 0);
  end_ns                       = read_i64(src + 8);
  std::uint16_t const name_len = read_u16(src + 16);
  name.assign(reinterpret_cast<char const*>(src + 18), name_len);
  return k_kernel_marker_header + name_len;
}

/**
 * @brief Decode one event from `src` into `out`.
 *
 * @return Number of bytes consumed.
 */
std::size_t decode_into(std::uint8_t const* src, cupti::kernel_event& out)
{
  return decode_kernel_or_marker(src, out.start_ns, out.end_ns, out.name);
}
std::size_t decode_into(std::uint8_t const* src, cupti::memcpy_event& out)
{
  out.start_ns = read_i64(src + 0);
  out.end_ns   = read_i64(src + 8);
  out.kind     = src[16];
  out.bytes    = read_u64(src + 17);
  return k_memcpy_size;
}
std::size_t decode_into(std::uint8_t const* src, cupti::marker_event& out)
{
  return decode_kernel_or_marker(src, out.start_ns, out.end_ns, out.name);
}
std::size_t decode_into(std::uint8_t const* src, cupti::mem_decompress_event& out)
{
  out.start_ns     = read_i64(src + 0);
  out.end_ns       = read_i64(src + 8);
  out.source_bytes = read_u64(src + 16);
  return k_mem_decompress_size;
}

/**
 * @brief Append events from `src[cursor..]` into `payload` starting at
 * `off`, advancing `cursor` and `off`. Stops at the first event whose
 * `size_in_shmem` would exceed the remaining capacity.
 *
 * @return True iff every remaining event was packed (source drained).
 *         False if the buffer filled first.
 */
template <class Vec>
bool pack_type(Vec const& src,
               std::size_t& cursor,
               std::uint8_t* payload,
               std::size_t& off,
               std::size_t cap,
               std::uint32_t& count)
{
  while (cursor < src.size()) {
    auto const& e = src[cursor];
    if (off + size_in_shmem(e) > cap) return false;
    off += encode_into(payload + off, e);
    ++count;
    ++cursor;
  }
  return true;
}

/**
 * @brief Decode `n` consecutive events from `payload[off..]` into the
 * back of `dst`, advancing `off`.
 */
template <class Vec>
void unpack_n(Vec& dst, std::uint8_t const* payload, std::size_t& off, std::uint32_t n)
{
  dst.reserve(dst.size() + n);
  for (std::uint32_t i = 0; i < n; ++i) {
    typename Vec::value_type e;
    off += decode_into(payload + off, e);
    dst.push_back(std::move(e));
  }
}

/**
 * @brief Walk a chunk and append its events to `dst`. Name bytes are
 * copied into owning `std::string` fields.
 */
void append_chunk(agg_slot const& slot, cupti::activity_records& dst)
{
  std::uint8_t const* const payload = slot.payload;
  std::size_t off                   = 0;
  unpack_n(dst.kernels, payload, off, slot.num_kernels);
  unpack_n(dst.memcopies, payload, off, slot.num_memcopies);
  unpack_n(dst.markers, payload, off, slot.num_markers);
  unpack_n(dst.mem_decompress, payload, off, slot.num_mem_decompress);
}

/**
 * @brief Copy a slot's scalar fields (per-stage timestamps, time
 * breakdown, counter values) into a fresh `per_rank_data`. The events
 * vectors are left empty; the caller fills them.
 *
 * The publisher writes these scalars once per run and only advances
 * `gen` afterward, so they are valid for any reader that has observed
 * the final chunk's gen advance.
 */
agg_protocol::per_rank_data read_slot_scalars(agg_slot const& slot)
{
  agg_protocol::per_rank_data d;
  std::copy(
    std::begin(slot.stage_starts_ns), std::end(slot.stage_starts_ns), d.stage_starts_ns.begin());
  std::copy(std::begin(slot.stage_ends_ns), std::end(slot.stage_ends_ns), d.stage_ends_ns.begin());
  d.breakdown_valid = slot.breakdown_valid != 0;
  if (d.breakdown_valid) {
    d.breakdown.in_memory_read_task_s = slot.tb_in_memory_read_task_s;
    d.breakdown.compute_kernel_s      = slot.tb_compute_kernel_s;
    d.breakdown.io_kernel_s           = slot.tb_io_kernel_s;
    d.breakdown.memcpy_s              = slot.tb_memcpy_s;
    d.breakdown.mem_decompress_s      = slot.tb_mem_decompress_s;
    d.breakdown.merged_io_activity_s  = slot.tb_merged_io_activity_s;
  }
  d.counters.reserve(slot.num_counters);
  for (std::uint32_t i = 0; i < slot.num_counters; ++i) {
    d.counters.emplace_back(slot.counters[i].name, slot.counters[i].value);
  }
  return d;
}

}  // namespace

agg_protocol agg_protocol::attach(char const* shm_name,
                                  std::size_t env_size,
                                  std::uint32_t total_ranks,
                                  std::uint32_t rank)
{
  if (rank >= total_ranks) {
    throw std::runtime_error{"agg_protocol::attach: rank " + std::to_string(rank) +
                             " >= total_ranks " + std::to_string(total_ranks)};
  }

  std::size_t const expected_size =
    sizeof(agg_header) + static_cast<std::size_t>(total_ranks) * sizeof(agg_slot);
  if (env_size != expected_size) {
    throw std::runtime_error{
      "agg_protocol::attach: GQE_BENCH_SHM_SIZE=" + std::to_string(env_size) +
      " disagrees with layout (header=" + std::to_string(sizeof(agg_header)) + " + " +
      std::to_string(total_ranks) + " * slot=" + std::to_string(sizeof(agg_slot)) + " = " +
      std::to_string(expected_size) + ")"};
  }

  int const fd = shm_open(shm_name, O_RDWR, 0);
  if (fd < 0) {
    int const err = errno;
    throw std::runtime_error{std::string{"agg_protocol::attach: shm_open(\""} + shm_name +
                             "\", O_RDWR) failed: " + errno_string(err)};
  }

  void* base                 = mmap(nullptr, env_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  int const saved_mmap_errno = errno;
  ::close(fd);
  if (base == MAP_FAILED) {
    throw std::runtime_error{std::string{"agg_protocol::attach: mmap("} + std::to_string(env_size) +
                             ", MAP_SHARED, \"" + shm_name +
                             "\") failed: " + errno_string(saved_mmap_errno)};
  }

  GQE_BENCH_LOG_INFO(
    "agg_protocol attached: rank={}/{} size={} name={}", rank, total_ranks, env_size, shm_name);
  return agg_protocol{
    std::unique_ptr<void, munmap_deleter>{base, munmap_deleter{env_size}}, rank, total_ranks};
}

agg_protocol::agg_protocol(std::unique_ptr<void, munmap_deleter> base,
                           std::uint32_t rank,
                           std::uint32_t total_ranks) noexcept
  : _base(std::move(base)), _rank(rank), _total_ranks(total_ranks)
{
  _expected_gen.assign(total_ranks, 1);
}

agg_slot* agg_protocol::slots() const noexcept
{
  auto* bytes = static_cast<std::byte*>(_base.get());
  return reinterpret_cast<agg_slot*>(bytes + sizeof(agg_header));
}

agg_slot& agg_protocol::slot_at(std::uint32_t r) noexcept { return slots()[r]; }
agg_slot const& agg_protocol::slot_at(std::uint32_t r) const noexcept { return slots()[r]; }

void agg_protocol::barrier() noexcept
{
  std::atomic_ref<std::uint32_t> ready{header()->ready_count};
  std::uint32_t const after = ready.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (after > _total_ranks) {
    // Sanity check: more ranks attached than `total_ranks`.
    GQE_BENCH_LOG_ERROR(
      "agg_protocol barrier: ready_count overshoot ({} > {})", after, _total_ranks);
  }

  while (ready.load(std::memory_order_acquire) < _total_ranks) {}
}

std::int32_t agg_protocol::requested_experiment_id() const noexcept
{
  return std::atomic_ref<std::int32_t>{header()->requested_experiment_id}.load(
    std::memory_order_acquire);
}

void agg_protocol::fill_scalar_fields(agg_slot& slot, run_context const& ctx) noexcept
{
  using ns_t       = std::chrono::nanoseconds;
  auto const tp_ns = [](auto tp) {
    return std::chrono::duration_cast<ns_t>(tp.time_since_epoch()).count();
  };
  for (std::size_t i = 0; i < stages.size(); ++i) {
    slot.stage_starts_ns[i] = tp_ns(ctx.stage_starts[i]);
    slot.stage_ends_ns[i] =
      slot.stage_starts_ns[i] + std::chrono::duration_cast<ns_t>(ctx.stage_durations[i]).count();
  }

  slot.breakdown_valid = ctx.breakdown_valid ? 1 : 0;
  if (ctx.breakdown_valid) {
    auto const& b                 = ctx.breakdown;
    slot.tb_in_memory_read_task_s = b.in_memory_read_task_s;
    slot.tb_compute_kernel_s      = b.compute_kernel_s;
    slot.tb_io_kernel_s           = b.io_kernel_s;
    slot.tb_memcpy_s              = b.memcpy_s;
    slot.tb_mem_decompress_s      = b.mem_decompress_s;
    slot.tb_merged_io_activity_s  = b.merged_io_activity_s;
  } else {
    slot.tb_in_memory_read_task_s = 0.0;
    slot.tb_compute_kernel_s      = 0.0;
    slot.tb_io_kernel_s           = 0.0;
    slot.tb_memcpy_s              = 0.0;
    slot.tb_mem_decompress_s      = 0.0;
    slot.tb_merged_io_activity_s  = 0.0;
  }

  // Counters: cap at k_max_counters; truncate names at k_max_counter_name_len-1.
  std::uint32_t n = 0;
  for (auto const& [name, value] : ctx.counter_values) {
    if (n >= k_max_counters) break;
    auto& entry              = slot.counters[n];
    std::size_t const to_cpy = std::min<std::size_t>(name.size(), k_max_counter_name_len - 1);
    std::memcpy(entry.name, name.data(), to_cpy);
    entry.name[to_cpy] = '\0';
    entry.value        = value;
    ++n;
  }
  slot.num_counters = n;
}

void agg_protocol::publish(cupti::activity_records activity, std::uint64_t& own_gen)
{
  if (_rank == 0) {
    _self_events = std::move(activity);
    return;
  }

  // Scalar fields were filled by `fill_scalar_fields` before this call;
  // only the per-chunk fields and payload are touched in the loop.
  agg_slot* const dst = &slots()[_rank];

  // Cursors over each event vector. Events are emitted in this fixed
  // order: kernels, memcopies, markers, mem_decompress.
  std::size_t kernels_cursor        = 0;
  std::size_t memcopies_cursor      = 0;
  std::size_t markers_cursor        = 0;
  std::size_t mem_decompress_cursor = 0;

  std::uint64_t this_chunk_gen = own_gen;

  // Loop until every event is published, even if zero events (still
  // emit one chunk with final=1 so rank 0's drain terminates).
  while (true) {
    std::uint8_t* const payload        = dst->payload;
    std::size_t const cap              = k_payload_bytes;
    std::size_t off                    = 0;
    std::uint32_t chunk_kernels        = 0;
    std::uint32_t chunk_memcopies      = 0;
    std::uint32_t chunk_markers        = 0;
    std::uint32_t chunk_mem_decompress = 0;

    // Each pack_type fills until its vector exhausts or the chunk fills.
    // The && chain short-circuits to the next type only when the current
    // one drains; `drained` is true iff every type drained in this chunk.
    bool const drained =
      pack_type(activity.kernels, kernels_cursor, payload, off, cap, chunk_kernels) &&
      pack_type(activity.memcopies, memcopies_cursor, payload, off, cap, chunk_memcopies) &&
      pack_type(activity.markers, markers_cursor, payload, off, cap, chunk_markers) &&
      pack_type(
        activity.mem_decompress, mem_decompress_cursor, payload, off, cap, chunk_mem_decompress);

    dst->num_kernels        = chunk_kernels;
    dst->num_memcopies      = chunk_memcopies;
    dst->num_markers        = chunk_markers;
    dst->num_mem_decompress = chunk_mem_decompress;
    dst->payload_size       = static_cast<std::uint32_t>(off);
    dst->last_chunk         = drained ? 1 : 0;

    // Release-store gen — synchronizes-with rank 0's acquire-load.
    std::atomic_ref<std::uint64_t>(dst->gen).store(this_chunk_gen, std::memory_order_release);

    if (drained) {
      own_gen = this_chunk_gen + 1;
      return;
    }

    // Wait for rank 0 to drain this chunk before overwriting the slot.
    std::atomic_ref<std::uint64_t> consumed{dst->consumed_gen};
    while (consumed.load(std::memory_order_acquire) != this_chunk_gen) {}
    ++this_chunk_gen;
  }
}

void agg_protocol::reset_run_state() noexcept
{
  _own_gen = 1;
  std::fill(_expected_gen.begin(), _expected_gen.end(), 1);
  _self_events.reset();
}

void agg_protocol::publish_self(run_context& ctx)
{
  fill_scalar_fields(slot_at(_rank), ctx);
  publish(std::move(ctx.activity_records), _own_gen);
}

agg_protocol::per_rank_data agg_protocol::drain_one_rank(std::uint32_t r)
{
  if (r == _rank) {
    per_rank_data d = read_slot_scalars(slot_at(r));
    if (_self_events.has_value()) {
      d.events = std::move(*_self_events);
      _self_events.reset();
    }
    return d;
  }

  agg_slot* const slot = &slots()[r];
  per_rank_data d;

  while (true) {
    std::atomic_ref<std::uint64_t> gen{slot->gen};
    while (gen.load(std::memory_order_acquire) != _expected_gen[r]) {}

    append_chunk(*slot, d.events);
    bool const final_chunk = slot->last_chunk != 0;

    // Ack: release-store consumed_gen so the publisher can reuse the slot.
    std::atomic_ref<std::uint64_t>(slot->consumed_gen)
      .store(_expected_gen[r], std::memory_order_release);
    ++_expected_gen[r];
    if (final_chunk) break;
  }

  // The publisher writes slot scalars before advancing `gen`, so the
  // final-chunk gen advance observed above implies the scalars are
  // visible to this thread.
  per_rank_data scalars = read_slot_scalars(*slot);
  scalars.events        = std::move(d.events);
  return scalars;
}

agg_protocol::gathered_run agg_protocol::gather_run()
{
  gathered_run g;
  g.per_rank.reserve(_total_ranks);

  for (std::uint32_t r = 0; r < _total_ranks; ++r) {
    g.per_rank.push_back(drain_one_rank(r));
  }

  // Cross-rank reduction: build and execute span the earliest start to
  // the latest end across all ranks; collect uses rank 0's value
  // unchanged.
  auto const& r0                                = g.per_rank[0];
  std::array<std::int64_t, stages.size()> mins  = r0.stage_starts_ns;
  std::array<std::int64_t, stages.size()> maxes = r0.stage_ends_ns;
  for (std::uint32_t r = 1; r < _total_ranks; ++r) {
    auto const& d = g.per_rank[r];
    for (std::size_t s = 0; s < stages.size(); ++s) {
      mins[s]  = std::min(mins[s], d.stage_starts_ns[s]);
      maxes[s] = std::max(maxes[s], d.stage_ends_ns[s]);
    }
  }
  auto const seconds = [](std::int64_t s, std::int64_t e) {
    return static_cast<double>(e - s) / 1e9;
  };
  std::size_t const b  = static_cast<std::size_t>(stage::build);
  std::size_t const ex = static_cast<std::size_t>(stage::execute);
  std::size_t const co = static_cast<std::size_t>(stage::collect);
  g.stages.build_s     = seconds(mins[b], maxes[b]);
  g.stages.execute_s   = seconds(mins[ex], maxes[ex]);
  g.stages.collect_s   = seconds(r0.stage_starts_ns[co], r0.stage_ends_ns[co]);
  g.stages.total_s     = g.stages.build_s + g.stages.execute_s;

  return g;
}

}  // namespace gqe_bench
