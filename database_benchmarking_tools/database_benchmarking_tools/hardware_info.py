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


class GpuInfo:
    def __init__(self):
        pynvml.nvmlInit()

    def __del__(self):
        pynvml.nvmlShutdown()

    def cuda_driver_version(self) -> str:
        version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        return str(version // 1000) + "." + str((version // 10) % 10)

    def device_product_name(self, gpu_id: int) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        return pynvml.nvmlDeviceGetName(handle)

    def gpu_cores(self, gpu_id: int) -> int:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        # This API call needs to check for error since it fails on WSL2
        try:
            return pynvml.nvmlDeviceGetNumGpuCores(handle)
        except pynvml.NVMLError_NotSupported:
            return None

    def max_memory_clock(self, gpu_id: int) -> int:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)

    def max_sm_clock(self, gpu_id: int) -> int:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)

    def pcie_link_generation(self, gpu_id: int) -> int:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        return pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)

    def system_driver_version(self) -> str:
        return pynvml.nvmlSystemGetDriverVersion()

    def total_ecc_errors(self, gpu_id: int) -> int:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
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
