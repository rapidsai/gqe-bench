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

import pynvml
from cuda.bindings import driver as cuda

# CUDA logical index -> UUID via the driver API. NVML's enumeration is
# independent of ``CUDA_VISIBLE_DEVICES``, so the NVML index can't be used.


def _uuid_for_cuda_index(cuda_index: int) -> str:
    (err,) = cuda.cuInit(0)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err}")
    err, dev = cuda.cuDeviceGet(cuda_index)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet({cuda_index}) failed: {err}")
    err, uuid = cuda.cuDeviceGetUuid(dev)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGetUuid({cuda_index}) failed: {err}")
    h = bytes(uuid.bytes).hex()
    return f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# Thin wrapper over NVML used to read the GPU properties stored in the
# `gpu_info` and `hw_info` dimension tables. Each accessor takes a CUDA device
# index and returns the corresponding NVML-reported value (product name, UUID,
# clocks, ECC counters, ...). NVML handles are resolved by UUID (via CUDA's
# logical-index-to-UUID mapping) so the result matches the device that
# `cudaSetDevice(gpu_id)` will bind in the engine. Distinct from the
# `GpuInfo` dataclass in `experiment.py`, which is the table mapping for the
# rows produced from these queries.
class GpuInfoQuery:
    def __init__(self):
        pynvml.nvmlInit()

    def __del__(self):
        pynvml.nvmlShutdown()

    def _handle_for_cuda_index(self, cuda_index: int):
        return pynvml.nvmlDeviceGetHandleByUUID(_uuid_for_cuda_index(cuda_index))

    def cuda_driver_version(self) -> str:
        version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        return str(version // 1000) + "." + str((version // 10) % 10)

    def device_product_name(self, gpu_id: int) -> str:
        handle = self._handle_for_cuda_index(gpu_id)
        return pynvml.nvmlDeviceGetName(handle)

    def device_uuid(self, gpu_id: int) -> str:
        return _uuid_for_cuda_index(gpu_id)

    def gpu_cores(self, gpu_id: int) -> int:
        handle = self._handle_for_cuda_index(gpu_id)
        # This API call needs to check for error since it fails on WSL2
        try:
            return pynvml.nvmlDeviceGetNumGpuCores(handle)
        except pynvml.NVMLError_NotSupported:
            return None

    def max_memory_clock(self, gpu_id: int) -> int:
        handle = self._handle_for_cuda_index(gpu_id)
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)

    def max_sm_clock(self, gpu_id: int) -> int:
        handle = self._handle_for_cuda_index(gpu_id)
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)

    def pcie_link_generation(self, gpu_id: int) -> int:
        handle = self._handle_for_cuda_index(gpu_id)
        return pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)

    def system_driver_version(self) -> str:
        return pynvml.nvmlSystemGetDriverVersion()

    def total_ecc_errors(self, gpu_id: int) -> int:
        handle = self._handle_for_cuda_index(gpu_id)
        try:
            corrected = pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                pynvml.NVML_AGGREGATE_ECC,
            )
            uncorrected = pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                pynvml.NVML_AGGREGATE_ECC,
            )
            return corrected + uncorrected
        except pynvml.NVMLError_NotSupported:
            return None


class CpuInfo:
    def __init__(self):
        self._info = {}

        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    self._info[key.strip()] = value.strip()

    def model_name(self) -> str:
        return self._info.get("model name", "Unknown")

    def cpu_mhz(self) -> float:
        return float(self._info.get("cpu MHz", -1))

    def cpu_physical_cores(self) -> int:
        return int(self._info.get("cpu cores", -1))
