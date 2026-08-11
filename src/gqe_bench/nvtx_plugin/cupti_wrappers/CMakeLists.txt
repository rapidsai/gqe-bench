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

add_library(cupti_wrappers_obj OBJECT cupti_activity.cpp cupti_range.cpp)
target_compile_features(cupti_wrappers_obj PRIVATE cxx_std_20)
# Warning set matches gqe/CMakeLists.txt for CXX targets.
target_compile_options(
  cupti_wrappers_obj
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Werror>
          $<$<COMPILE_LANGUAGE:CXX>:-Wimplicit-fallthrough>
          $<$<COMPILE_LANGUAGE:CXX>:-Wsuggest-override>)
target_include_directories(
  cupti_wrappers_obj
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
          ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/CUPTI/include)
target_link_libraries(cupti_wrappers_obj PRIVATE CUDA::cupti CUDA::cuda_driver)

if(GQE_BUILD_PLUGIN_TESTS)
  add_subdirectory(tests)
endif()
