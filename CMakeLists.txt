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

cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

option(GQE_BUILD_NVTX_PLUGIN
       "Build the gqe_bench NVTX plugin shared library (libgqe_bench.so)" ON)
option(
  GQE_FETCH_ENGINE
  "Fetch and build gqe_node_manager/gqe_task_manager/gqe-cli from the gqe repo"
  ON)
option(GQE_ENABLE_PUBLIC_BUILD
       "Enable building for external user (affects the fetched GQE engine)" ON)
option(GQE_BUILD_PLUGIN_TESTS
       "Build the plugin C++ unit tests (requires nvcc + fetches gtest)" OFF)

if(GQE_FETCH_ENGINE OR GQE_BUILD_PLUGIN_TESTS)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80-real" "90-real" "100-real" "120")
  endif()
  project(gqe_bench LANGUAGES C CXX CUDA)
  set(CMAKE_CUDA_STANDARD 20)
else()
  project(gqe_bench LANGUAGES C CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Install layout: wheel.install-dir (pyproject.toml) prefixes installs with
# gqe_bench/_artifacts/. BINDIR/LIBDIR default to bin/lib so gqe's
# $ORIGIN/../${LIBDIR} INSTALL_RPATH resolves bin -> sibling lib.

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  if(GQE_FETCH_ENGINE)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
  endif()
endif()

include(cmake/CPM.cmake)

if(GQE_BUILD_PLUGIN_TESTS)
  enable_testing()
endif()

# GQE engine + gqe-cli.
#
# GQE's CMakeLists install()s the executables with
# $ORIGIN/../${GQE_INSTALL_LIBDIR} INSTALL_RPATH that resolves the sibling
# arrow/nvcomp install. GQE's rapids_cmake_install_lib_dir respects the
# CMAKE_INSTALL_BINDIR/LIBDIR, which resolve to bin/lib from GNUInstallDirs'
# defaults (pulled in transitively by the GQE build). For editable-install
# parity we also redirect executable build output to the source tree and force
# BUILD_WITH_INSTALL_RPATH so the source-tree copy resolves sibling libs without
# a wheel install roundtrip.
if(GQE_FETCH_ENGINE)
  # nvCOMP selection belongs to GQE; forward only the variables a caller
  # explicitly set. CPM passes OPTIONS as normal variables, which shadow the
  # cache defaults GQE declares in cmake/nvcomp.cmake, so forwarding an unset
  # variable would blank GQE's default rather than leave it alone.
  set(_gqe_options "CMAKE_POSITION_INDEPENDENT_CODE ON"
                   "GQE_ENABLE_PUBLIC_BUILD ${GQE_ENABLE_PUBLIC_BUILD}")
  foreach(_var GQE_NVCOMP_TAG GQE_NVCOMP_SOURCE_DIR GQE_NVCOMP_LIB_DIR
               GQE_NVCOMP_PUBLIC_VERSION)
    if(DEFINED ${_var})
      list(APPEND _gqe_options "${_var} ${${_var}}")
    endif()
  endforeach()

  if(DEFINED GQE_SOURCE_DIR)
    message(STATUS "Using GQE from local source directory: ${GQE_SOURCE_DIR}")
    cpmaddpackage(NAME libgqe SOURCE_DIR ${GQE_SOURCE_DIR} OPTIONS
                  ${_gqe_options})
  else()
    set(GQE_GIT_REPOSITORY
        "https://gitlab-master.nvidia.com/Devtech-Compute/gqe.git"
        CACHE STRING "Git repository URL for gqe")
    set(GQE_GIT_TAG
        "ff00b01f7db7065f1243b434b496495c1d52eeff"
        CACHE STRING "Git tag/commit for gqe")

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
      ${_gqe_options})
  endif()

  # gqe-cli: GQE builds it via cargo into ${GQE_RUST_TARGET_DIR}/release/ but
  # does not install it. Install rule here lands it in CMAKE_INSTALL_BINDIR
  # alongside the C++ executables; cargo build runs as part of ALL so the file
  # exists by install time.
  install(PROGRAMS ${GQE_RUST_TARGET_DIR}/release/gqe-cli
          DESTINATION ${CMAKE_INSTALL_BINDIR})

endif()

# NVTX plugin (LD_PRELOAD'd libgqe_bench.so that observes GQE's NVTX ranges via
# the CUPTI Callback API and writes timing rows into the .db3).
#
# Added after the engine package: rapids_logger promotes its spdlog/fmt imported
# targets to global scope, and acquiring spdlog here beforehand would create
# those targets in this directory and make that promotion fail.
if(GQE_BUILD_NVTX_PLUGIN)
  find_package(CUDAToolkit REQUIRED COMPONENTS cupti)

  # The plugin formats its records with spdlog to match the engine's pattern.
  # GQE resolves spdlog inside its own directory scope, so the imported target
  # is not visible here; find it unless the engine has already put it in scope.
  if(NOT TARGET spdlog::spdlog)
    find_package(spdlog REQUIRED)
  endif()

  cpmaddpackage(
    NAME
    sqlite3
    VERSION
    3.53.4
    URL
    "https://www.sqlite.org/2024/sqlite-amalgamation-3450100.zip"
    DOWNLOAD_ONLY
    YES)

  if(sqlite3_ADDED)
    add_library(sqlite3 STATIC ${sqlite3_SOURCE_DIR}/sqlite3.c)
    target_include_directories(sqlite3 PUBLIC ${sqlite3_SOURCE_DIR})
    target_compile_definitions(sqlite3 PRIVATE SQLITE_THREADSAFE=1
                                               SQLITE_ENABLE_WAL=1)
  endif()

  add_subdirectory(src/gqe_bench/nvtx_plugin)
endif()

# All files generated into the source tree, and the values they are generated
# from. Included last: it references targets defined by the plugin subdirectory
# and variables set by the engine package.
include(cmake/codegen.cmake)
