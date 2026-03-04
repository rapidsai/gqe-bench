#!/usr/bin/env python3
#
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

"""
Tests for JSON configuration parsing in param_sweep_config.py
"""

from argparse import Namespace

import pytest

from gqe_bench.param_sweep_config import (
    BENCHMARK_CONFIG_DEFAULTS,
    QUERY_CONFIG_FIELDS,
    config_to_args,
    get_query_execution_params,
)


class TestConfigToArgs:
    """Tests for config_to_args function."""

    def test_returns_namespace(self):
        """config_to_args returns a Namespace object."""
        config = {
            "dataset": "/path/to/dataset",
            "plan": "/path/to/plan",
            "solution": "/path/to/solution",
        }
        result = config_to_args(config)
        assert isinstance(result, Namespace)

    def test_applies_defaults(self):
        """config_to_args applies BENCHMARK_CONFIG_DEFAULTS for missing fields."""
        config = {
            "dataset": "/path/to/dataset",
            "plan": "/path/to/plan",
            "solution": "/path/to/solution",
        }
        result = config_to_args(config)
        assert result.partitions == BENCHMARK_CONFIG_DEFAULTS["partitions"]
        assert result.workers == BENCHMARK_CONFIG_DEFAULTS["workers"]
        assert result.repeat == BENCHMARK_CONFIG_DEFAULTS["repeat"]

    def test_overrides_defaults(self):
        """config_to_args overrides defaults with config values."""
        config = {
            "dataset": "/path/to/dataset",
            "plan": "/path/to/plan",
            "solution": "/path/to/solution",
            "partitions": [16, 32],
            "repeat": 10,
        }
        result = config_to_args(config)
        assert result.partitions == [16, 32]
        assert result.repeat == 10

    def test_stores_query_overrides(self):
        """config_to_args stores query_overrides for per-query lookups."""
        config = {
            "dataset": "/path/to/dataset",
            "plan": "/path/to/plan",
            "solution": "/path/to/solution",
            "query_overrides": [{"queries": ["1"], "partitions": [1]}],
        }
        result = config_to_args(config)
        assert result.query_overrides == [{"queries": ["1"], "partitions": [1]}]

    def test_empty_query_overrides_when_not_present(self):
        """config_to_args sets empty list when no query_overrides in config."""
        config = {
            "dataset": "/path/to/dataset",
            "plan": "/path/to/plan",
            "solution": "/path/to/solution",
        }
        result = config_to_args(config)
        assert result.query_overrides == []


def _make_args(**overrides):
    """Helper to create args Namespace with defaults."""
    defaults = {
        "partitions": [1, 2, 4, 8],
        "workers": [1],
        "read_use_filter_pruning": [False],
        "read_use_overlap_mtx": [True],
        "read_use_zero_copy": [False],
        "filter_use_like_shift_and": [True],
        "join_use_hash_map_cache": [True],
        "join_use_unique_keys": [True],
        "join_use_perfect_hash": [False],
        "join_use_mark_join": [False],
        "aggregation_use_perfect_hash": [True],
        "query_overrides": [],
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestGetQueryExecutionParams:
    """Tests for get_query_execution_params function."""

    def test_returns_all_query_config_fields(self):
        """Returns dict with all QUERY_CONFIG_FIELDS."""
        args = _make_args()
        result = get_query_execution_params(args, "1")
        for field in QUERY_CONFIG_FIELDS:
            assert field in result

    def test_uses_args_when_no_overrides(self):
        """Uses args values when no query overrides exist."""
        args = _make_args(partitions=[16, 32], workers=[2])
        result = get_query_execution_params(args, "1")
        assert result["partitions"] == [16, 32]
        assert result["workers"] == [2]

    def test_applies_single_override(self):
        """Applies override when query matches, replacing args value."""
        args = _make_args(
            partitions=[1, 2, 4, 8],
            query_overrides=[{"queries": ["1", "2"], "partitions": [16]}],
        )
        result = get_query_execution_params(args, "1")
        assert result["partitions"] == [16]

    def test_non_matching_query_uses_args(self):
        """Non-matching query uses args values."""
        args = _make_args(
            partitions=[1, 2],
            query_overrides=[{"queries": ["3"], "partitions": [16]}],
        )
        result = get_query_execution_params(args, "1")
        assert result["partitions"] == [1, 2]

    def test_merges_multiple_matching_overrides(self):
        """Multiple matching overrides have their lists merged."""
        args = _make_args(
            partitions=[1],
            query_overrides=[
                {"queries": ["1", "2", "3"], "partitions": [2]},
                {"queries": ["1"], "partitions": [4]},
            ],
        )
        result = get_query_execution_params(args, "1")
        # Overrides replace args: first [2] + second [4] = [2, 4]
        assert result["partitions"] == [2, 4]

    def test_merge_keeps_unique_values(self):
        """Merging overrides keeps only unique values."""
        args = _make_args(
            partitions=[1, 2],
            query_overrides=[
                {"queries": ["1"], "partitions": [2, 3]},
                {"queries": ["1"], "partitions": [3, 4]},
            ],
        )
        result = get_query_execution_params(args, "1")
        # Overrides only: [2, 3] + [3, 4] -> [2, 3, 4]
        assert result["partitions"] == [2, 3, 4]

    def test_override_only_affects_specified_fields(self):
        """Override only affects fields it specifies."""
        args = _make_args(
            partitions=[1, 2],
            workers=[1],
            query_overrides=[{"queries": ["1"], "partitions": [4]}],
        )
        result = get_query_execution_params(args, "1")
        assert result["partitions"] == [4]  # Override replaces args
        assert result["workers"] == [1]  # Unchanged (no override)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
