/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <utility/config.hpp>
#include <utility/utility.hpp>

#include <cassert>
#include <cstdint>
#include <cuda/std/optional>
#include <cuda/std/tuple>

/**
 * @brief A shared memory write buffer.
 *
 * The write buffer optimizes writing to device memory. It buffers the writes of
 * a warp in a shared memory buffer. When the buffer is full, the writes are
 * flushed to device memory.
 *
 * The advantages are:
 *  - Less contention on the atomic offset variable.
 *  - Write alignment, if the initial offset is aligned. Writes are aligned to
 * `column_capacity`.
 *  - Write coalescing.
 *
 * TODO: Use C++17 template fold expressions to generalize the number of table
 * columns.
 */
template <typename T>
class write_buffer_op {
 public:
  static constexpr uint32_t leader         = gqe_python::utility::warp_leader;
  static constexpr std::size_t num_columns = 1;  /// Width of output table.
  static constexpr std::size_t column_capacity =
    gqe_python::utility::output_cache_size;  // Per-column shared memory buffer capacity.
  static constexpr std::size_t buffer_capacity =
    column_capacity * gqe_python::utility::warp_size *
    num_columns;  /// Total shared memory buffer capacity.

  /**
   * A temporary storage area that must be allocated in shared memory.
   */
  struct storage_t {
    cuda::atomic<uint32_t, cuda::thread_scope_block> buffer_offset[gqe_python::utility::warp_size];
    T buffer[buffer_capacity];
  };

  /**
   * @brief A new write buffer.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   *
   * @arg[in] storage A shared memory temporary storage area. Must be allocated
   * by the callee, e.g., using `__shared__ storage_t s;`.
   * @arg[in,out] output_table The output table to which tuples will be written.
   * @arg[in,out] output_table_offset A write offset in the output table, that
   * indicates the next empty row.
   */
  __device__ write_buffer_op(
    storage_t* storage,
    T* output_table,
    cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> output_table_offset)
    : storage(storage), output_table(output_table), output_table_offset(output_table_offset)
  {
    assert(gqe::utility::warp_size == warpSize);

    if (gqe_python::utility::thread_rank() == leader) {
      storage->buffer_offset[gqe_python::utility::warp_id()].store(0, cuda::memory_order_relaxed);
    }

    __syncwarp();
  }

  /**
   * @brief Buffering write of a tuple to the output table.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   *
   * @arg[in] tuple The tuple to write. If a thread has no tuple, it should pass
   * `nullopt`.
   */

  __device__ void write(cuda::std::optional<T> value) noexcept
  {
    int32_t do_fill = value.has_value();

    while (true) {
      if (do_fill) {
        uint32_t offset = storage->buffer_offset[gqe_python::utility::warp_id()].fetch_add(
          1, cuda::memory_order_relaxed);

        if (offset < column_capacity) {
          // Only store the row index.
          get_element(offset, 0) = value.value();
          do_fill                = false;
        }
      }

      int32_t do_flush = __any_sync(gqe_python::utility::warp_full_mask, do_fill);

      if (do_flush) {
        flush_all();
        __syncwarp();
      } else {
        break;
      }
    }
  }

  /**
   * @brief Flush the remaining write buffer of warp to the output table.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   * @post Warp must be synchronized before next call to `write()` (e.g., with
   * `__syncwarp()`).
   */
  __device__ void flush() noexcept
  {
    __syncwarp();

    uint32_t current_offset = 1;  // non-zero for clarity

    if (gqe_python::utility::thread_rank() == leader) {
      current_offset =
        storage->buffer_offset[gqe_python::utility::warp_id()].load(cuda::memory_order_relaxed);
    }
    current_offset = __shfl_sync(gqe_python::utility::warp_full_mask, current_offset, leader);

    if (current_offset == 0) { return; }

    flush_impl(current_offset);
  }

 private:
  /**
   * @brief Flush the write buffer of a warp to the output table.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   * @post Warp must be synchronized before next call to `write()` (e.g., with
   * `__syncwarp()`).
   *
   * @param[in] buffer_size The number of rows to flush.
   *
   * @details Implements the flush.
   */
  __device__ void flush_impl(uint32_t buffer_size) noexcept
  {
    cudf::size_type target_offset = 0;

    if (gqe_python::utility::thread_rank() == leader) {
      target_offset = output_table_offset.fetch_add(buffer_size, cuda::memory_order_relaxed);
    }

    target_offset = __shfl_sync(gqe_python::utility::warp_full_mask, target_offset, leader);

    for (int32_t i = gqe_python::utility::thread_rank(); i < buffer_size; i += warpSize) {
      output_table[target_offset + i] = get_element(i, 0);
    }
  }

  /**
   * @brief Flush the *whole* write buffer of warp to the output table.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   * @pre Warp must be synchronized before call (e.g., with `__syncwarp()`).
   */
  __device__ void flush_all() noexcept
  {
    if (gqe_python::utility::thread_rank() == leader) {
      storage->buffer_offset[gqe_python::utility::warp_id()].store(0, cuda::memory_order_relaxed);
    }

    flush_impl(column_capacity);
  }

  /**
   * @brief Get an element in the buffer.
   *
   * @param[in] column_idx The element's column index relative the output table.
   * @param[in] row_idx The element's row index in the write buffer.
   *
   * @return A reference to the element in the write buffer.
   */
  __device__ T& get_element(int32_t row_idx, int32_t column_idx) const noexcept
  {
    return storage->buffer[num_columns * column_capacity * gqe_python::utility::warp_id() +
                           column_capacity * column_idx + row_idx];
  }

  storage_t* storage;  /// Storage array containing the shared memory write buffer.
  T* output_table;     /// Pointers to the columns of the output table.
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device>
    output_table_offset;  /// Current row offset in the output table.
};
