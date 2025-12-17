# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
This module defines the data type objects used in GQE.
"""

from abc import ABC, abstractmethod

import gqe.lib


class DataType(ABC):
    @abstractmethod
    def _to_cpp(self) -> gqe.lib.DataType:
        pass


class Int64(DataType):
    def _to_cpp(self):
        return gqe.lib.DataType(gqe.lib.TypeId.int64)


class Float64(DataType):
    def _to_cpp(self):
        return gqe.lib.DataType(gqe.lib.TypeId.float64)
