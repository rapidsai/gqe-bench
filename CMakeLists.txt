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

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70" "80" "90")
endif()

project(gqe-python LANGUAGES CXX CUDA)

include(cmake/CPM.cmake)
CPMAddPackage(
  NAME libgqe
  GIT_REPOSITORY ssh://git@gitlab-master.nvidia.com:12051/Devtech-Compute/gqe.git
  GIT_TAG 4405f49441a12f319f3c2c96abe8d2607492b526
  # We could also use the local GQE copy for development
  # SOURCE_DIR <path-to-GQE-repo>
  OPTIONS "CMAKE_POSITION_INDEPENDENT_CODE ON"
)

rapids_find_package(pybind11 REQUIRED)
pybind11_add_module(lib gqe/lib.cpp)
target_link_libraries(lib PUBLIC gqe)
install(TARGETS lib LIBRARY DESTINATION gqe)
