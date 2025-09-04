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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70-real" "80-real" "90-real" "100-real" "120")
endif()

project(gqe-python LANGUAGES CXX CUDA)

include(cmake/CPM.cmake)

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
    set(GQE_GIT_TAG "53b533242d2124ddccc3d9056d4ab1da31c4032e")
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
pybind11_add_module(lib gqe/lib.cpp)
target_link_libraries(lib PUBLIC gqe)
install(TARGETS lib LIBRARY DESTINATION gqe)
