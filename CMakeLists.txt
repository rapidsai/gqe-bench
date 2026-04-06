# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

option(GQE_ENABLE_PUBLIC_BUILD "Enable building for external user" ON)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
  set(env{CCACHE_BASEDIR} ${CMAKE_SOURCE_DIR})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-U__ARM_NEON")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80-real" "90-real" "100-real" "120")
endif()

project(gqe_bench LANGUAGES CXX CUDA)

include(cmake/CPM.cmake)

set(GQE_NVCOMP_LIB_DIR
    ""
    CACHE
      PATH
      "Optional path to prebuilt nvCOMP install (must contain cmake config files)."
)

# CMake variable for gqe-nvcomp tag
set(GQE_NVCOMP_TAG
    "614b388a8466a0020d9ee7e6fd53f69fc5790ac2"
    CACHE STRING "Git tag for gqe-nvcomp")
set(GQE_NVCOMP_SOURCE_DIR
    ""
    CACHE PATH
          "Local source directory for gqe-nvcomp (overrides GIT_TAG if set)")

# needed for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler options Temporarily disable warnings and errors that are causing
# issues with nvcomp. add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
# add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
# add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>")
# add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Werror>")

if(DEFINED GQE_SOURCE_DIR)
  message(STATUS "Using GQE from local source directory: ${GQE_SOURCE_DIR}")
  cpmaddpackage(
    NAME
    libgqe
    SOURCE_DIR
    ${GQE_SOURCE_DIR}
    OPTIONS
    "CMAKE_POSITION_INDEPENDENT_CODE ON"
    "GQE_ENABLE_SUBSTRAIT_PRODUCER ${GQE_ENABLE_SUBSTRAIT_PRODUCER}"
    "GQE_NVCOMP_TAG ${GQE_NVCOMP_TAG}"
    "GQE_NVCOMP_SOURCE_DIR ${GQE_NVCOMP_SOURCE_DIR}"
    "GQE_NVCOMP_LIB_DIR ${GQE_NVCOMP_LIB_DIR}"
    "GQE_ENABLE_PUBLIC_BUILD ${GQE_ENABLE_PUBLIC_BUILD}")
else()
  if(NOT DEFINED GQE_GIT_REPOSITORY)
    set(GQE_GIT_REPOSITORY
        "https://gitlab-master.nvidia.com/Devtech-Compute/gqe.git")
  endif()
  if(NOT DEFINED GQE_GIT_TAG)
    set(GQE_GIT_TAG "2d6a06e4c5929c09efe33f18f6ba2904d3d9b56d")
  endif()

  message(STATUS "Using GQE from repository: ${GQE_GIT_REPOSITORY}")
  message(STATUS "Using GQE tag: ${GQE_GIT_TAG}")

  cpmaddpackage(
    NAME
    libgqe
    GIT_REPOSITORY
    ${GQE_GIT_REPOSITORY}
    GIT_TAG
    ${GQE_GIT_TAG}
    OPTIONS
    "CMAKE_POSITION_INDEPENDENT_CODE ON"
    "GQE_ENABLE_SUBSTRAIT_PRODUCER ${GQE_ENABLE_SUBSTRAIT_PRODUCER}"
    "GQE_NVCOMP_TAG ${GQE_NVCOMP_TAG}"
    "GQE_NVCOMP_SOURCE_DIR ${GQE_NVCOMP_SOURCE_DIR}"
    "GQE_NVCOMP_LIB_DIR ${GQE_NVCOMP_LIB_DIR}"
    "GQE_ENABLE_PUBLIC_BUILD ${GQE_ENABLE_PUBLIC_BUILD}")
endif()

rapids_find_package(pybind11 REQUIRED)

# PREPEND our lib path to ensure it takes priority over conda's nvcomp This
# assumes the layout: /conda/envs/gqe/site-packages/gqe/lib.so and
# /conda/envs/gqe/site-packages/lib/libnvcomp.so
list(PREPEND CMAKE_BUILD_RPATH "$ORIGIN/../lib")
list(PREPEND CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# Add path to libnvcomp.so to the build and install RPATHs
if(GQE_NVCOMP_LIB_DIR)
  list(PREPEND CMAKE_BUILD_RPATH "${GQE_NVCOMP_LIB_DIR}/lib")
  list(PREPEND CMAKE_INSTALL_RPATH "${GQE_NVCOMP_LIB_DIR}/lib")
endif()

pybind11_add_module(
  lib
  gqe_bench/lib.cpp
  gqe_bench/benchmark/hardcoded/src/tpch/q10/fused_probes.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q10/fused_probes_task.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q10/sort_limit_task.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q10/unique_key_inner_join.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q10/unique_key_inner_join_task.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q13/filter_orders_task.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q13/groupjoin_task.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q13/kernels.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q18/groupby_task.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q22/task.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q22/fused_project_filter.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q22/mark_join.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q16/join_task.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q16/aggregate_task.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q16/fused_filter_join.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q21/mark_join_task.cu
  gqe_bench/benchmark/hardcoded/src/tpch/q21/mark_join.cu)

# Include the GQE build directory to get the git revision header
target_include_directories(
  lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gqe_bench/benchmark/hardcoded/include
              ${libgqe_BINARY_DIR})
target_compile_options(
  lib
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
          $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
target_link_libraries(lib PUBLIC gqe)
install(TARGETS lib LIBRARY DESTINATION gqe_bench)
