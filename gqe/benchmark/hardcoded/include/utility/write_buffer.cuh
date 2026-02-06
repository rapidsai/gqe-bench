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

namespace gqe_python {
namespace utility {

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
 * TODO: Fix the column alignment when using different data types. E.g.,
 * `write_buffer_op<int32_t, int64_t>`.
 */
template <typename... Value>
class write_buffer_op {
 public:
  static constexpr int32_t num_columns = sizeof...(Value);  /// Width of output table.
  static constexpr std::size_t op_pack_bytes =
    (0 + ... + sizeof(Value));  /// The size of the template op pack in bytes.
  static constexpr std::size_t column_capacity =
    output_cache_size;  /// Per-column shared memory buffer capacity.
  static constexpr std::size_t buffer_capacity_bytes =
    column_capacity * max_num_warps * op_pack_bytes;  /// Total shared memory buffer capacity.

  /**
   * A temporary storage area that must be allocated in shared memory.
   */
  struct storage_t {
    cuda::atomic<uint32_t, cuda::thread_scope_block> buffer_offset[max_num_warps];
    uint8_t buffer[buffer_capacity_bytes];
  };

  /**
   * @brief A new write buffer.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   *
   * @arg[in] storage A shared memory temporary storage area. Must be allocated
   * by the callee, e.g., using `__shared__ storage_t s;`.
   * @arg[in,out] output_table_offset A write offset in the output table, that
   * indicates the next empty row.
   * @arg[in,out] output_columns The output columns to which tuples will be written.
   */
  __device__ write_buffer_op(
    storage_t* storage,
    cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> output_table_offset,
    Value*... output_columns)
    : storage(storage),
      output_columns(cuda::std::make_tuple(output_columns...)),
      output_table_offset(output_table_offset)
  {
    if (thread_rank() == warp_leader) {
      storage->buffer_offset[warp_id()].store(0, cuda::memory_order_relaxed);
    }

    __syncwarp();
  }

  /**
   * @brief Buffering write of a tuple to the output table.
   *
   * @pre Warp-synchronous function; must be called by all threads.
   *
   * @arg[in] tuple The tuple to write. If a thread has no tuple, it should pass
   * `cuda::std::nullopt`.
   */
  __device__ void write(cuda::std::optional<cuda::std::tuple<Value...>> tuple) noexcept
  {
    int32_t do_fill = tuple.has_value();

    while (true) {
      if (do_fill) {
        uint32_t offset =
          storage->buffer_offset[warp_id()].fetch_add(1, cuda::memory_order_relaxed);

        if (offset < column_capacity) {
          buffer_elements<0, 0>(offset, tuple.value());
          do_fill = false;
        }
      }

      int32_t do_flush = __any_sync(warp_full_mask, do_fill);

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

    if (thread_rank() == warp_leader) {
      current_offset = storage->buffer_offset[warp_id()].load(cuda::memory_order_relaxed);
    }
    current_offset = __shfl_sync(warp_full_mask, current_offset, warp_leader);

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

    if (thread_rank() == warp_leader) {
      target_offset = output_table_offset.fetch_add(buffer_size, cuda::memory_order_relaxed);
    }

    target_offset = __shfl_sync(warp_full_mask, target_offset, warp_leader);

    for (int32_t i = thread_rank(); i < buffer_size; i += warpSize) {
      write_elements<0, 0>(target_offset + i, i);
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
    if (thread_rank() == warp_leader) {
      storage->buffer_offset[warp_id()].store(0, cuda::memory_order_relaxed);
    }

    flush_impl(column_capacity);
  }

  /**
   * @brief Recursively store elements in the shared memory buffer.
   *
   * The recursion is unrolled at compile-time.
   *
   * @param[in] column_idx The element's column index relative the output table.
   * @param[in] column_offset The column offset in bytes in the buffer.
   * @param[in] row_idx The element's row index in the write buffer.
   * @param[in] values The element values to be buffered.
   */
  template <int32_t column_idx, size_t column_offset>
  __device__ void buffer_elements(int32_t row_idx, cuda::std::tuple<Value...> values)
  {
    using T = typename cuda::std::tuple_element<column_idx, cuda::std::tuple<Value...>>::type;

    auto buffer_idx =
      column_capacity * op_pack_bytes * warp_id() + column_offset + row_idx * sizeof(T);
    auto slot = reinterpret_cast<T*>(&storage->buffer[buffer_idx]);

    *slot = cuda::std::get<column_idx>(values);

    if constexpr (column_idx + 1 < num_columns) {
      // `column_offset` is needed because the value types can have different sizes. E.g.,
      // `write_buffer_op<int32_t, int64_t>`. Thus, we can't obtain the offset by `num_columns *
      // sizeof(T)`.
      buffer_elements<column_idx + 1, column_offset + column_capacity * sizeof(T)>(row_idx, values);
    }
  }

  /**
   * @brief Recursively write back elements from the buffer to the output columns.
   *
   * The recursion is unrolled at compile-time.
   *
   * @param[in] column_idx The element's column index relative the output table.
   * @param[in] column_offset The column offset in bytes in the buffer.
   * @param[in] output_idx The element's row index in the output columns.
   * @param[in] row_idx The element's row index in the write buffer.
   */
  template <int32_t column_idx, size_t column_offset>
  __device__ void write_elements(cudf::size_type output_idx, int32_t row_idx)
  {
    using T = typename cuda::std::tuple_element<column_idx, cuda::std::tuple<Value...>>::type;

    auto output_column = cuda::std::get<column_idx>(output_columns);
    auto buffer_idx =
      column_capacity * op_pack_bytes * warp_id() + column_offset + row_idx * sizeof(T);
    auto slot = reinterpret_cast<T*>(&storage->buffer[buffer_idx]);

    output_column[output_idx] = *slot;

    if constexpr (column_idx + 1 < num_columns) {
      write_elements<column_idx + 1, column_offset + column_capacity * sizeof(T)>(output_idx,
                                                                                  row_idx);
    }
  }

  storage_t* storage;  /// Storage array containing the shared memory write buffer.
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device>
    output_table_offset;                       /// Current row offset in the output table.
  cuda::std::tuple<Value*...> output_columns;  /// Pointers to the columns of the output table.
};

}  // namespace utility
}  // namespace gqe_python
