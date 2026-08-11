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

add_subdirectory(cupti_wrappers)

# env.hpp and stages.hpp, included by the sources below, are generated into the
# build tree by cmake/codegen.cmake, which also puts them on this target's
# include path.
add_library(
  gqe_bench SHARED
  plugin.cpp
  observers.cpp
  capturers.cpp
  env.cpp
  nvtx_injection.cpp
  nvtx_domain_filter.cpp
  agg_protocol.cpp
  run_writer.cpp)
target_compile_features(gqe_bench PRIVATE cxx_std_20)
# Warning set matches gqe/CMakeLists.txt for CXX targets — keep in sync.
target_compile_options(
  gqe_bench
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Werror>
          $<$<COMPILE_LANGUAGE:CXX>:-Wimplicit-fallthrough>
          $<$<COMPILE_LANGUAGE:CXX>:-Wsuggest-override>)
target_include_directories(
  gqe_bench PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/CUPTI/include)
target_link_libraries(
  gqe_bench PRIVATE cupti_wrappers_obj CUDA::cupti CUDA::cuda_driver
                    CUDA::cudart sqlite3 spdlog::spdlog)

install(TARGETS gqe_bench LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Size probe: a tiny host executable that includes agg_protocol.hpp and writes
# sizeof(agg_header) and sizeof(agg_slot) into _agg_protocol_sizes.py, the
# authoritative source of segment-size constants for
# gqe_bench.recording._build_plugin_env. cmake/codegen.cmake runs it.
add_executable(agg_protocol_size_probe agg_protocol_size_probe.cpp)
target_compile_features(agg_protocol_size_probe PRIVATE cxx_std_20)
target_include_directories(
  agg_protocol_size_probe
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/CUPTI/include)
