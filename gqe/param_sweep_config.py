#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
JSON configuration file support for TPC-H parameter sweep.

This module provides functionality to load experiment configurations from JSON files,
with support for global parameters and query-specific overrides.
"""

import json
import sys
from argparse import Namespace
from typing import Any


# Fields that can be overridden per-query, with their default values
QUERY_CONFIG_DEFAULTS = {
    "partitions": [1, 2, 4, 8],
    "workers": [1],
    "read_use_filter_pruning": [False],
    "read_use_overlap_mtx": [False, True],
    "read_use_zero_copy": [False, True],
    "filter_use_like_shift_and": [False, True],
    "join_use_hash_map_cache": [False, True],
    "join_use_unique_keys": [True],
    "join_use_perfect_hash": [False, True],
    "join_use_mark_join": [False, True],
    "aggregation_use_perfect_hash": [False, True],
}

# Set of field names that can be overridden per-query
QUERY_CONFIG_FIELDS = set(QUERY_CONFIG_DEFAULTS.keys())

# Default values matching the CLI argument defaults
BENCHMARK_CONFIG_DEFAULTS = {
    "output": None,
    "quiet": False,
    "queries": None,
    "identifier_type": None,
    "row_groups": None,
    "query_source": "both",
    "compression_format": ["none"],
    "load_all_data": None,
    "metrics": None,
    "storage_kind": ["numa_pinned_memory"],
    "multiprocess": False,
    "repeat": 6,
    "query_timeout": 1800,
    "data_timeout": 10800,
    "boost_pool_size": None,
    "sandboxing": False,
    "verify_results": True,
    **QUERY_CONFIG_DEFAULTS,
}

# Required fields that must be present in the JSON config
REQUIRED_FIELDS = ["dataset", "plan", "solution"]


def load_json_config(path: str) -> dict:
    """
    Load and validate a JSON configuration file.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If required fields are missing.
    """
    with open(path, "r") as f:
        config = json.load(f)

    # Check for required fields
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Missing required fields in config: {missing}")

    return config


def config_to_args(config: dict) -> Namespace:
    """
    Convert a JSON configuration dictionary to an argparse Namespace object.

    This allows the rest of the code to use the configuration in the same way
    as command-line arguments.

    Args:
        config: Configuration dictionary loaded from JSON.

    Returns:
        Namespace object with configuration values.
    """
    args_dict = {}

    # Start with defaults
    args_dict.update(BENCHMARK_CONFIG_DEFAULTS)

    # Override with config values (excluding query_overrides which is handled separately)
    for key, value in config.items():
        if key != "query_overrides":
            args_dict[key] = value

    # Store query overrides for per-query lookups
    args_dict["query_overrides"] = config.get("query_overrides", [])

    return Namespace(**args_dict)


def _merge_unique(existing: list, new: list) -> list:
    """Merge two lists, keeping only unique values while preserving order."""
    seen = set(existing)
    result = list(existing)
    for item in new:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def check_cli_overrides(argv: list[str]) -> None:
    """
    Check if CLI arguments other than --json/-j were provided and warn the user.

    Args:
        argv: Command line arguments (typically sys.argv).
    """
    # Find the index of --json or -j to skip its value
    i = 0
    other_args = []
    while i < len(argv):
        arg = argv[i]
        if arg in ("--json", "-j"):
            # Skip the --json flag and its value
            i += 2
            continue
        if arg.startswith("-"):
            other_args.append(arg)
        i += 1

    if other_args:
        print(
            f"Warning: The following CLI arguments are ignored when using --json: {other_args}",
            file=sys.stderr,
        )


def get_query_execution_params(args: Namespace, query_str: str) -> dict:
    """
    Get execution parameters for a query, merging any JSON overrides with args defaults.

    This function handles the logic of checking for JSON config overrides and
    falling back to args values. It returns a dictionary with all execution
    parameters needed for itertools.product in the parameter sweep.

    Args:
        args: Namespace object with configuration.
        query_str: The query string (e.g., "1", "2_fused_filter").

    Returns:
        Dictionary with keys: workers, partitions, read_use_overlap_mtx,
        join_use_hash_map_cache, read_use_zero_copy, join_use_unique_keys,
        join_use_perfect_hash, join_use_mark_join, read_use_filter_pruning,
        filter_use_like_shift_and, aggregation_use_perfect_hash.
    """
    query_overrides = getattr(args, "query_overrides", [])

    # Collect all matching override entries for this query
    matching_overrides = [
        entry for entry in query_overrides if query_str in entry.get("queries", [])
    ]

    params = {}
    for field in QUERY_CONFIG_FIELDS:
        # Check if any matching override provides this field
        override_values = [
            entry[field] for entry in matching_overrides if field in entry
        ]

        if override_values:
            # Use only override values, merging multiple overrides together
            result = list(override_values[0])
            for values in override_values[1:]:
                result = _merge_unique(result, values)
            params[field] = result
        else:
            # Fall back to args value
            params[field] = getattr(args, field)

    return params
