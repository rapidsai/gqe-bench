#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import csv
from enum import Enum
import sqlite3


# source: https://docs.nvidia.com/nsight-systems/2021.5/nsys-exporter/exported_data.html#cuda-copykind-enum
class CUDA_MEMCPY_KIND(Enum):
    CUDA_MEMCPY_KIND_UNKNOWN = 0
    CUDA_MEMCPY_KIND_HTOD = 1
    CUDA_MEMCPY_KIND_DTOH = 2
    CUDA_MEMCPY_KIND_HTOA = 3
    CUDA_MEMCPY_KIND_ATOH = 4
    CUDA_MEMCPY_KIND_ATOA = 5
    CUDA_MEMCPY_KIND_ATOD = 6
    CUDA_MEMCPY_KIND_DTOA = 7
    CUDA_MEMCPY_KIND_DTOD = 8
    CUDA_MEMCPY_KIND_HTOH = 9
    CUDA_MEMCPY_KIND_PTOP = 10
    CUDA_MEMCPY_KIND_UVM_HTOD = 11
    CUDA_MEMCPY_KIND_UVM_DTOH = 12
    CUDA_MEMCPY_KIND_UVM_DTOD = 13


# source: https://docs.nvidia.com/nsight-systems/2021.5/nsys-exporter/exported_data.html#cuda-memorykind-enum
class CUDA_MEMOPR_MEMORY_KIND(Enum):
    CUDA_MEMOPR_MEMORY_KIND_PAGEABLE = 0
    CUDA_MEMOPR_MEMORY_KIND_PINNED = 1
    CUDA_MEMOPR_MEMORY_KIND_DEVICE = 2
    CUDA_MEMOPR_MEMORY_KIND_ARRAY = 3
    CUDA_MEMOPR_MEMORY_KIND_MANAGED = 4
    CUDA_MEMOPR_MEMORY_KIND_DEVICE_STATIC = 5
    CUDA_MEMOPR_MEMORY_KIND_MANAGED_STATIC = 6
    CUDA_MEMOPR_MEMORY_KIND_UNKNOWN = 7


# source: https://docs.nvidia.com/nsight-systems/2021.5/nsys-exporter/exported_data.html#nvtx-eventtype-values
class NVTX_EVENT_TYPE(Enum):
    NVTX_CATEGORY = 33
    NVTX_MARK = 34
    NVTX_THREAD = 39
    NVTX_PUSH_POP_RANGE = 59
    NVTX_START_END_RANGE = 60
    NVTX_DOMAIN_CREATE = 75
    NVTX_DOMAIN_DESTROY = 76


GQE_DOMAIN_NAME = "GQE"
GQE_IN_MEMORY_READ_TASK_NVTX_RANGE = "in_memory_read_task"


def merge_ranges(ranges: list) -> list:
    ranges.sort(key=lambda x: x[0])
    merged = []
    for range in ranges:
        if not merged or merged[-1][1] < range[0]:
            merged.append(list(range))
        else:
            merged[-1][1] = max(merged[-1][1], range[1])
    return merged


def kernel_time(connection, nvtx_range_glob, args):
    headers = ["nvtx_domain", "kernel_count", "total_kernel_time_ms"]
    sql_str = f"""
    SELECT 
    nvtx_string.value,
    COUNT(kernel.correlationId) AS kernel_count,
    ROUND(SUM(kernel.end - kernel.start) / 1000000.0, 3) AS total_kernel_time_ms
    
    FROM NVTX_EVENTS nvtx
    JOIN StringIds AS nvtx_string
        ON nvtx_string.id = nvtx.textId -- get nvtx domain names
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME runtime_activity 
        ON (runtime_activity.end <= nvtx.end
            AND runtime_activity.start >= nvtx.start) -- get runtime call within nvtx range
    JOIN CUPTI_ACTIVITY_KIND_KERNEL kernel
        ON (kernel.correlationId = runtime_activity.correlationId) -- get kernels
    JOIN StringIds kernel_string
        ON (kernel.demangledName = kernel_string.id) -- get kernel names
    WHERE 
        (nvtx.eventType = {NVTX_EVENT_TYPE.NVTX_PUSH_POP_RANGE.value} 
        OR nvtx.eventType = {NVTX_EVENT_TYPE.NVTX_START_END_RANGE.value}
        ) -- push-pop and start-end ranges
        AND nvtx_string.value GLOB "{nvtx_range_glob}" 
        {f"AND kernel_string.value NOT GLOB '{args.exclude_kernel_glob}'" 
            if args.exclude_kernel_glob 
            else ""
        }
    GROUP BY 
        nvtx_string.value, nvtx.start, nvtx.end 
    ORDER BY 
        nvtx.start
    """
    rows = list(connection.execute(sql_str))
    return [dict(zip(headers, row)) for row in rows]


def htod_copy_time_sum(connection, nvtx_range_glob, args) -> list[dict[str, float]]:
    headers = ["nvtx_domain", "total_memcpy_time_ms"]
    sql_str = f"""
    SELECT 
        nvtx_string.value,
        ROUND(SUM(memcpy.end - memcpy.start) / 1000000.0, 3) AS total_memcpy_time_ms
    FROM NVTX_EVENTS nvtx
    JOIN StringIds AS nvtx_string
        ON nvtx_string.id = nvtx.textId -- get nvtx domain names
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME runtime_activity 
        ON (runtime_activity.start >= nvtx.start
            AND runtime_activity.end <= nvtx.end
            AND ) -- get runtime call within nvtx range
    JOIN CUPTI_ACTIVITY_KIND_MEMCPY memcpy
        ON (memcpy.correlationId = runtime_activity.correlationId) -- get memcpy calls
    WHERE 
        (nvtx.eventType = {NVTX_EVENT_TYPE.NVTX_PUSH_POP_RANGE.value} 
        OR nvtx.eventType = {NVTX_EVENT_TYPE.NVTX_START_END_RANGE.value}
        )  -- push-pop and start-end ranges
        AND memcpy.copyKind = {CUDA_MEMCPY_KIND.CUDA_MEMCPY_KIND_HTOD.value} -- HtoD copies
        AND memcpy.srcKind = {CUDA_MEMOPR_MEMORY_KIND.CUDA_MEMOPR_MEMORY_KIND_PINNED.value} -- src is Pinned memory
        AND nvtx_string.value GLOB "{nvtx_range_glob}" 
    GROUP BY 
        nvtx_string.value, nvtx.start, nvtx.end 
    ORDER BY 
        nvtx.start
    """
    rows = list(connection.execute(sql_str))
    return [dict(zip(headers, row)) for row in rows]


def htod_copy_size(connection, nvtx_range_glob, args):
    headers = ["nvtx_domain", "total_memcpy_size_MiB"]
    sql_str = f"""
    SELECT 
        nvtx_string.value,
        ROUND(SUM(memcpy.bytes) / 1024.0 / 1024.0, 3) AS total_memcpy_size_MiB
    FROM NVTX_EVENTS nvtx
    JOIN StringIds AS nvtx_string
        ON nvtx_string.id = nvtx.textId -- get nvtx domain names
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME runtime_activity 
        ON (runtime_activity.start >= nvtx.start
            AND runtime_activity.end <= nvtx.end) -- get runtime call within nvtx range
    JOIN CUPTI_ACTIVITY_KIND_MEMCPY memcpy
        ON (memcpy.correlationId = runtime_activity.correlationId) -- get memcpy calls
    WHERE 
        (nvtx.eventType = {NVTX_EVENT_TYPE.NVTX_PUSH_POP_RANGE.value} 
        OR nvtx.eventType = {NVTX_EVENT_TYPE.NVTX_START_END_RANGE.value}
        )  -- push-pop and start-end ranges
        AND memcpy.copyKind = {CUDA_MEMCPY_KIND.CUDA_MEMCPY_KIND_HTOD.value} -- HtoD copies
        AND memcpy.srcKind = {CUDA_MEMOPR_MEMORY_KIND.CUDA_MEMOPR_MEMORY_KIND_PINNED.value} -- src is Pinned memory
        AND nvtx_string.value GLOB "{nvtx_range_glob}" 
    GROUP BY 
        nvtx_string.value, nvtx.start, nvtx.end 
    ORDER BY 
        nvtx.start
    """
    rows = list(connection.execute(sql_str))
    return [dict(zip(headers, row)) for row in rows]


def read_time_effective(connection, nvtx_range_glob, args) -> list[dict[str, float]]:

    # GQE domain name and tasks nvtx ranges are stored in the text field
    # See https://docs.nvidia.com/nsight-systems/2021.5/nsys-exporter/exported_data.html#difference-between-text-and-textid-columns

    gqe_domain_id = connection.execute(
        f"""
    SELECT domainId from NVTX_EVENTS WHERE text = "{GQE_DOMAIN_NAME}"
    """
    ).fetchone()[0]
    query = f"""
    SELECT 
        nvtx_string.value,
        GROUP_CONCAT(
            ROUND(nvtx_read_task.start / 1000000.0, 3) || "," || ROUND(nvtx_read_task.end / 1000000.0, 3),
            ";"
        ) AS read_tasks_ranges
    FROM NVTX_EVENTS nvtx
    JOIN StringIds AS nvtx_string
        ON nvtx_string.id = nvtx.textId -- get nvtx domain names
    JOIN NVTX_EVENTS nvtx_read_task
        ON (nvtx_read_task.start >= nvtx.start
            AND nvtx_read_task.end <= nvtx.end) -- get read task within nvtx range
    WHERE 
        (nvtx.eventType = {NVTX_EVENT_TYPE.NVTX_PUSH_POP_RANGE.value} 
        OR nvtx.eventType = {NVTX_EVENT_TYPE.NVTX_START_END_RANGE.value}
        )  -- push-pop and start-end ranges
        AND nvtx_string.value GLOB "{nvtx_range_glob}"
        AND nvtx_read_task.domainId = {gqe_domain_id}
        AND nvtx_read_task.text = "{GQE_IN_MEMORY_READ_TASK_NVTX_RANGE}" -- GQE ranges are stored in text field
    GROUP BY 
        nvtx_string.value, nvtx.start, nvtx.end
    ORDER BY 
        nvtx_read_task.start
    """
    rows = list(connection.execute(query))
    merged_rows = []
    for nvtx_domain, read_tasks_ranges in rows:
        read_task_ranges = [
            tuple(map(float, range.split(",")))
            for range in read_tasks_ranges.split(";")
        ]
        merged_ranges = merge_ranges(read_task_ranges)
        merged_rows.append(
            {
                "nvtx_domain": nvtx_domain,
                "total_read_task_time_ms": sum(
                    range[1] - range[0] for range in merged_ranges
                ),
            }
        )
    return merged_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform analysis on the supplied nvtx range from an nsys SQLite file"
    )

    subparsers = parser.add_subparsers(dest="tool", required=True)
    parser.add_argument("sqlite", help="Path to SQLite file for analysis")
    parser.add_argument("nvtx_range_glob", help="Glob pattern of NVTX range to analyze")
    parser.add_argument("-o", "--output", default=None, help="Output to csv file")

    kernel_time_tool_parser = subparsers.add_parser(
        "kernel_time",
        help="Sum up the time for kernels launched in the supplied nvtx range",
        epilog="Example:\n  python nsys_analysis.py kernel_time "
        '--exclude_kernel_glob "*fused_concatenate*" nsys-file.sqlite "*Run Q13*"',
    )

    kernel_time_tool_parser.add_argument(
        "--exclude_kernel_glob",
        default=None,
        help="Specify glob pattern for kernels which should be excluded from analysis",
    )

    kernel_time_tool_parser.set_defaults(program=kernel_time)

    io_analysis_parser = subparsers.add_parser(
        "io",
        help="Perform IO analysis",
        epilog='Example:\n  python nsys_analysis.py io nsys-file.sqlite "%Run Q13%" --analysis_type htod_copy_time_sum',
    )
    io_analysis_parser.add_argument(
        "--analysis_type",
        default="read_time_effective",
        choices=["htod_copy_time_sum", "htod_copy_size", "read_time_effective"],
        help=(
            "Type of IO analysis to perform. Options: "
            "htod_copy_time_sum - Total time spent copying host pinned column data to device (HtoD). "
            "htod_copy_size - Total number of host pinned column bytes transferred during in-memory read. "
            "read_time_effective - Effective end-to-end time for in-memory read tasks."
        ),
    )
    io_analysis_parser.set_defaults(program=read_time_effective)

    args = parser.parse_args()
    if args.command == "io":
        if args.analysis_type == "htod_copy_time_sum":
            args.program = htod_copy_time_sum
        elif args.analysis_type == "htod_copy_size":
            args.program = htod_copy_size
        elif args.analysis_type == "read_time_effective":
            args.program = read_time_effective

    conn = sqlite3.connect(args.sqlite)
    rows = args.program(conn, args.nvtx_range_glob, args)
    if args.csv:
        with open(args.csv, "w") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        print(rows)
