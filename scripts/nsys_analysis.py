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
import sqlite3

def kernel_time(connection, nvtx_range_glob, csv_output, args):

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
        (nvtx.eventType = 59 OR nvtx.eventType = 60) -- push-pop and start-end ranges
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
    headers = ["Name", "Num_Kernels", "Total_kernel_time(ms)"]
    if csv_output:
        with open(csv_output, 'x') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
    else:
        for row in rows:
            print(dict(zip(headers, row)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform analysis on the supplied nvtx range from an nsys SQLite file")
    
    subparsers = parser.add_subparsers(dest="tool", required=True)
    parser.add_argument("sqlite", help="Path to SQLite file for analysis")
    parser.add_argument("nvtx_range_glob", help="Glob pattern of NVTX range to analyze")
    parser.add_argument('-o', '--output', default=None, help="Output to csv file")

    kernel_time_tool_parser = subparsers.add_parser(
        "kernel_time", 
        help="Sum up the time for kernels launched in the supplied nvtx range", 
        epilog="Example:\n  python nsys_analysis.py kernel_time " \
        "--exclude_kernel_glob \"*fused_concatenate*\" nsys-file.sqlite \"*Run Q13*\"" )
    
    kernel_time_tool_parser.add_argument(
        '--exclude_kernel_glob',
        default=None, 
        help="Specify glob pattern for kernels which should be excluded from analysis")
    
    kernel_time_tool_parser.set_defaults(program=kernel_time)
    
    args = parser.parse_args()
    with sqlite3.connect(args.sqlite) as connection:
        args.program(connection, args.nvtx_range_glob, args.output, args)

