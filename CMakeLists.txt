# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
  set(env{CCACHE_BASEDIR} ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-U__ARM_NEON")

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80-real" "90-real" "100-real" "120")
endif()

project(gqe-python LANGUAGES CXX CUDA)

include(cmake/CPM.cmake)

# needed for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler options
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Werror>")

if(DEFINED GQE_SOURCE_DIR)
  message(STATUS "Using GQE from local source directory: ${GQE_SOURCE_DIR}")
  CPMAddPackage(
    NAME libgqe
    SOURCE_DIR ${GQE_SOURCE_DIR}
    OPTIONS "CMAKE_POSITION_INDEPENDENT_CODE ON"
  )
else()
  
  if(NOT DEFINED GQE_GIT_REPOSITORY)
    set(GQE_GIT_REPOSITORY "ssh://git@gitlab-master.nvidia.com:12051/Devtech-Compute/gqe.git")
  endif()

  if(NOT DEFINED GQE_GIT_TAG)
    set(GQE_GIT_TAG "7d55dd68a507dcf9d1347a9c7310aa3044d1585b")
  endif()
  
  message(STATUS "Using GQE from repository: ${GQE_GIT_REPOSITORY}")
  message(STATUS "Using GQE tag: ${GQE_GIT_TAG}")
  
  CPMAddPackage(
    NAME libgqe
    GIT_REPOSITORY ${GQE_GIT_REPOSITORY}
    GIT_TAG ${GQE_GIT_TAG}
    OPTIONS "CMAKE_POSITION_INDEPENDENT_CODE ON"
  )
endif()

rapids_find_package(pybind11 REQUIRED)

pybind11_add_module(lib
  gqe/lib.cpp
  gqe/benchmark/hardcoded/src/tpch/q10/fused_probes.cu
  gqe/benchmark/hardcoded/src/tpch/q10/fused_probes_task.cu
  gqe/benchmark/hardcoded/src/tpch/q10/sort_limit_task.cu
  gqe/benchmark/hardcoded/src/tpch/q10/unique_key_inner_join.cu
  gqe/benchmark/hardcoded/src/tpch/q10/unique_key_inner_join_task.cu
  gqe/benchmark/hardcoded/src/tpch/q13/filter_orders_task.cu
  gqe/benchmark/hardcoded/src/tpch/q13/groupjoin_task.cu
  gqe/benchmark/hardcoded/src/tpch/q13/kernels.cu
  gqe/benchmark/hardcoded/src/tpch/q22/task.cu
  gqe/benchmark/hardcoded/src/tpch/q22/fused_project_filter.cu
  gqe/benchmark/hardcoded/src/tpch/q22/mark_join.cu
  gqe/benchmark/hardcoded/src/tpch/q16/join_task.cu
  gqe/benchmark/hardcoded/src/tpch/q16/aggregate_task.cu
  gqe/benchmark/hardcoded/src/tpch/q16/fused_filter_join.cu
)

# Include the GQE build directory to get the git revision header
target_include_directories(lib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/gqe/benchmark/hardcoded/include
  ${libgqe_BINARY_DIR}
)
target_compile_options(lib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
)
target_link_libraries(lib PUBLIC gqe)
install(TARGETS lib LIBRARY DESTINATION gqe)
