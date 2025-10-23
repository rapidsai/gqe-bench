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

#include <gqe/context_reference.hpp>
#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/utility/error.hpp>

#include <cuda/functional>
#include <cuda/std/cstddef>

#include <cuco/bloom_filter.cuh>

#include <cstddef>
#include <cstdlib>

namespace gqe_python {
namespace utility {

/**
 * @brief Get the bloom filter number of blocks based on total number of bytes that bloom filter
 * takes up.
 *
 * @tparam filter_type
 * @param[in] filter_bytes The total number of bytes that the bloom filter takes up.
 */
template <typename filter_type>
cuco::extent<std::size_t> get_bloom_filter_blocks(std::size_t filter_bytes)
{
  std::size_t num_sub_filters =
    filter_bytes * filter_type::words_per_block / (sizeof(typename filter_type::word_type));

  return num_sub_filters;
}

/**
 * @brief Find the grid size based on block size and kernel.
 *
 * @tparam block_size
 * @tparam KernelType
 * @param kernel
 */
template <typename KernelType>
int find_grid_size(KernelType kernel, int block_size)
{
  int dev_id{-1};
  GQE_CUDA_TRY(cudaGetDevice(&dev_id));

  int num_sms{-1};
  GQE_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  int max_active_blocks{-1};
  GQE_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0));

  int grid_size = max_active_blocks * num_sms;

  return grid_size;
}

/**
 * @brief Rank of a thread within its warp.
 */
__device__ inline int thread_rank() noexcept { return threadIdx.x % warp_size; };

/**
 * @brief The warp identifier.
 */
__device__ inline int warp_id() noexcept { return threadIdx.x / warp_size; }

/**
 * @brief Number of warps in a thread block.
 */
__device__ inline int num_warps() noexcept { return blockDim.x / warp_size; };

/**
 * @brief Concatenate a list of tasks into one task. If there is only one task, return it directly.
 *
 * @param[in] ctx_ref The context in which the current task is running in.
 * @param[in] task_id Globally unique identifier of the task.
 * @param[in] stage_id Stage of the current task.
 * @param[in] tasks The list of tasks to be concatenated.
 */

inline std::shared_ptr<gqe::task> concatenate_tasks(gqe::context_reference ctx_ref,
                                                    int32_t& task_id,
                                                    int32_t stage_id,
                                                    std::vector<std::shared_ptr<gqe::task>> tasks)
{
  std::shared_ptr<gqe::task> concatenated_task;
  if (tasks.size() == 1) {
    concatenated_task = tasks[0];
  } else {
    concatenated_task = std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, tasks);
    task_id++;
  }
  return concatenated_task;
}

}  // namespace utility
}  // namespace gqe_python
