# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Builds the CUPTI wrapper unit tests. Only included when GQE_BUILD_PLUGIN_TESTS
# is ON; that option also enables CUDA as a language at the project level so
# this directory can compile a trivial `.cu` kernel.
#
# GTest is fetched once via CPM and cached. libgtest_main provides main().

cpmaddpackage(
  NAME
  googletest
  VERSION
  1.16.0
  GITHUB_REPOSITORY
  google/googletest
  GIT_TAG
  6910c9d9165801d8827d628cb72eb7ea9dd538c5 # v1.16.0 — pinned to commit, not tag
  OPTIONS
  "INSTALL_GTEST OFF"
  "BUILD_GMOCK OFF")

add_executable(cupti_test cupti_test.cpp test_kernel.cu)
target_compile_features(cupti_test PRIVATE cxx_std_20)
set_target_properties(cupti_test PROPERTIES CUDA_STANDARD 20)
# Warning set matches gqe/CMakeLists.txt for both CXX and CUDA languages.
target_compile_options(
  cupti_test
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall>
          $<$<COMPILE_LANGUAGE:CXX>:-Werror>
          $<$<COMPILE_LANGUAGE:CXX>:-Wimplicit-fallthrough>
          $<$<COMPILE_LANGUAGE:CXX>:-Wsuggest-override>
          $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>
          $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Werror>
          $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
target_include_directories(cupti_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(
  cupti_test PRIVATE cupti_wrappers_obj CUDA::cupti CUDA::cuda_driver
                     CUDA::cudart GTest::gtest GTest::gtest_main)
add_test(NAME cupti_test COMMAND cupti_test)
set_tests_properties(cupti_test PROPERTIES TIMEOUT 120 LABELS "cupti;gpu")
