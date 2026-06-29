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

# gqe_bench.lib statically links gqe's substrait .pb.cc files, which register
# substrait/*.proto descriptors in the global protobuf descriptor pool at
# module load time. pyarrow's libarrow_substrait.so registers the same
# descriptors and aborts the process on duplicate registration. pyarrow only
# loads libarrow_substrait when `pyarrow.substrait` is imported, and the only
# importer in pyarrow is pyarrow._dataset which wraps the import in a
# try/except ImportError. Mask the module before loading lib so the import
# fails cleanly and libarrow_substrait is never loaded into this process.
# TODO: Remove once we migrate to CLI. Also, monitor this whenever pyarrow updates in our environment.

import os
import sys as _sys

_sys.modules.setdefault("pyarrow.substrait", None)
_sys.modules.setdefault("pyarrow._substrait", None)

# Ensure CUDA enumerates devices by PCI bus ID, matching nvidia-smi ordering.
# Without this, CUDA defaults to FASTEST_FIRST which can cause
# CUDA_VISIBLE_DEVICES indices (set from nvidia-smi queries) to select the
# wrong GPU. Set in both binaries: the node manager propagates to spawned task
# managers via environ, but the task manager also sets it for standalone use.
# Reference:
# https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
#
# Must run before `gqe_bench.lib` is imported, because importing the C++
# extension can trigger CUDA driver initialization, after which this variable
# is no longer read.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from .catalog import Catalog  # noqa: E402
from .execute import Context, MultiProcessContext  # noqa: E402
from .relation import read  # noqa: E402
