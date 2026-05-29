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

from uuid import UUID

import pynvml
from cuda.bindings import runtime as cudart

# NVML reports/accepts GPU UUIDs as `"GPU-<uuid>"` (and `"MIG-..."` for MIG
# instances, which this module does not support).
# But UUID from cudaGetDeviceProperties doesn't have the prefix.
# The prefix is stripped and re-added at the NVML boundary so callers only deal in `uuid.UUID`.
_NVML_GPU_PREFIX = "GPU-"


# NVML-based wrapper that reads the GPU properties used to populate `gpu_info`.
# Per-device accessors take an opaque NVML handle from `handle_by_uuid` (or
# `handle_by_nvml_index`); prefer UUID-based lookup since NVML and CUDA-runtime
# index spaces can disagree under `CUDA_VISIBLE_DEVICES` or MIG.
class GpuInfoQuery:
    def __init__(self):
        pynvml.nvmlInit()

    def __del__(self):
        pynvml.nvmlShutdown()

    def cuda_driver_version(self) -> str:
        version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        return str(version // 1000) + "." + str((version // 10) % 10)

    def device_count(self) -> int:
        """Return the number of GPUs visible to NVML in this process."""
        return pynvml.nvmlDeviceGetCount()

    def handle_by_nvml_index(self, nvml_index: int) -> pynvml.c_nvmlDevice_t:
        """Return the NVML handle for the GPU at the given NVML index."""
        return pynvml.nvmlDeviceGetHandleByIndex(nvml_index)

    def handle_by_uuid(self, uuid: UUID) -> pynvml.c_nvmlDevice_t:
        """Return the NVML handle for the GPU with the given UUID.
        Use this to look up the physical device in NVML regardless of `CUDA_VISIBLE_DEVICES`.
        """
        return pynvml.nvmlDeviceGetHandleByUUID(f"{_NVML_GPU_PREFIX}{uuid}")

    def device_product_name(self, handle) -> str:
        return pynvml.nvmlDeviceGetName(handle)

    def device_uuid(self, handle) -> UUID:
        nvml_uuid = pynvml.nvmlDeviceGetUUID(handle)
        if not nvml_uuid.startswith(_NVML_GPU_PREFIX):
            raise ValueError(f"Unsupported NVML UUID form: {nvml_uuid!r}")
        return UUID(nvml_uuid.removeprefix(_NVML_GPU_PREFIX))

    def gpu_cores(self, handle) -> int:
        # This API call needs to check for error since it fails on WSL2
        try:
            return pynvml.nvmlDeviceGetNumGpuCores(handle)
        except pynvml.NVMLError_NotSupported:
            return None

    def max_memory_clock(self, handle) -> int:
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)

    def max_sm_clock(self, handle) -> int:
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)

    def pcie_link_generation(self, handle) -> int:
        return pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)

    def system_driver_version(self) -> str:
        return pynvml.nvmlSystemGetDriverVersion()

    def total_ecc_errors(self, handle) -> int:
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


def cuda_device_uuid(cuda_index: int) -> UUID:
    """Return the UUID of the GPU at the given CUDA device index, as
    reported by `cudaGetDeviceProperties`. Equal to `device_uuid` on the
    same physical device.
    """
    err, prop = cudart.cudaGetDeviceProperties(cuda_index)
    if err != cudart.cudaError_t.cudaSuccess:
        _, msg = cudart.cudaGetErrorString(err)
        raise RuntimeError(f"cudaGetDeviceProperties({cuda_index}) failed: {msg.decode()}")
    return UUID(bytes=bytes(prop.uuid.bytes))


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
