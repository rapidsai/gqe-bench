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

import logging
import os
import re
from typing import NamedTuple

import pynvml
from cuda.bindings import driver as cuda

logger = logging.getLogger(__name__)

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

    def cpu_sockets(self) -> int | None:
        """Number of physical CPU packages, or None when not reported.

        /proc/cpuinfo repeats "physical id" once per logical CPU, so the socket
        count is the number of distinct values it takes. Parsed here rather than
        from self._info, which keeps only the last value of each key. The field
        is absent on some architectures, notably arm64.
        """
        sockets = set()
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    key, _, value = line.partition(":")
                    if key.strip() == "physical id":
                        sockets.add(value.strip())
        except OSError:
            return None
        return len(sockets) or None


def parse_cpu_list(text: str) -> set[int]:
    """Parse a kernel CPU or node list such as ``0-3,8`` into a set of integers."""
    result = set()
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        result.update(range(int(first), int(last or first) + 1))
    return result


class NumaNode(NamedTuple):
    index: int
    memory_total_mb: int
    cpus: frozenset[int]


# Matches the bind policy that `numa_maps` prints for a mapping, such as `bind:0` or `bind:0-1`.
_BIND_POLICY = re.compile(r"^bind:([0-9,-]+)$")


class NumaTopology:
    """Reads the NUMA topology of the host and the NUMA binding of the calling process.

    The topology comes from sysfs. The binding is read from the effective state of the process
    rather than parsed from a `numactl` command line, so a binding set by any launcher is seen.

    The binding has the following limits.

    - The CPU binding is per node. A node counts as bound if the process may run on any of its
      CPUs, so `taskset -c 0-3` and `numactl --cpunodebind=0` both yield node 0. A cpuset cgroup
      limit on CPUs is included, because the kernel applies it to the affinity mask.
    - The memory binding recognizes only the `bind` policy, as set by `numactl --membind`. Other
      policies such as `interleave` and `preferred` are reported as every node, the same as an
      unbound process.
    - A cpuset cgroup limit on memory nodes, such as `docker run --cpuset-mems`, does not change
      the memory policy and is not seen.
    - The binding describes the calling process. A child process inherits it, but a thread that
      later sets its own affinity, or a memory range bound with `mbind`, is not seen.

    `nodes` scans sysfs on every call. Pass its result to `cpu_nodes` and `memory_nodes` so that
    the topology and the binding describe the same snapshot.
    """

    def __init__(
        self,
        sysfs_root: str = "/sys/devices/system/node",
        proc_root: str = "/proc",
    ):
        self._sysfs_root = sysfs_root
        self._proc_root = proc_root

    def nodes(self) -> list[NumaNode]:
        """Scan sysfs for the nodes present on this host, sorted by index.

        Returns an empty list if the system is not NUMA-aware. A node whose `meminfo` or `cpulist`
        cannot be read or parsed is skipped with a warning.
        """
        try:
            entries = os.listdir(self._sysfs_root)
        except OSError:
            return []

        nodes = []
        for entry in entries:
            match = re.fullmatch(r"node(\d+)", entry)
            if match is None:
                continue
            node_dir = os.path.join(self._sysfs_root, entry)
            try:
                with open(os.path.join(node_dir, "meminfo")) as meminfo:
                    memory_total_kb = next(
                        int(fields[3])
                        for fields in (line.split() for line in meminfo)
                        if len(fields) >= 4 and fields[2] == "MemTotal:"
                    )
                with open(os.path.join(node_dir, "cpulist")) as cpulist:
                    cpus = parse_cpu_list(cpulist.read())
            except (OSError, StopIteration, ValueError) as error:
                # StopIteration means that `meminfo` has no MemTotal line.
                logger.warning(
                    "Skipping NUMA node %s because %s cannot be parsed: %r",
                    match.group(1),
                    node_dir,
                    error,
                )
                continue
            nodes.append(NumaNode(int(match.group(1)), memory_total_kb // 1024, frozenset(cpus)))
        return sorted(nodes)

    def cpu_nodes(self, nodes: list[NumaNode]) -> list[int]:
        """Return the nodes that own a CPU the calling process may run on.

        `nodes` is the result of `nodes()`. Returns an empty list if the platform has no affinity
        API.
        """
        try:
            affinity = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            return []
        return [node.index for node in nodes if node.cpus & affinity]

    def memory_nodes(self, nodes: list[NumaNode]) -> list[int]:
        """Return the nodes the calling process may allocate memory from.

        `nodes` is the result of `nodes()`. A process without a `bind` policy may allocate from
        any node, so every node in `nodes` is returned.

        The policy is taken from the first mapping in `numa_maps`. That mapping is the executable
        image, which no allocator binds individually. It therefore shows the process-wide policy
        even when the process has bound later mappings itself.
        """
        all_nodes = [node.index for node in nodes]
        try:
            with open(os.path.join(self._proc_root, "self", "numa_maps")) as numa_maps:
                first_mapping = numa_maps.readline().split()
        except OSError:
            return all_nodes
        if len(first_mapping) >= 2:
            match = _BIND_POLICY.match(first_mapping[1])
            if match is not None:
                return sorted(parse_cpu_list(match.group(1)))
        return all_nodes
