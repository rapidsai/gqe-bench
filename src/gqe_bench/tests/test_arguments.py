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
Tests for JSON5 configuration and CLI argument parsing.

Usage:
    pytest gqe_bench/tests/test_arguments.py -v
"""

import json
import logging
from argparse import Namespace
from pathlib import Path

import pytest

from gqe_bench.arguments import (
    FIELDS,
    QUERY_FIELDS,
    config_to_args,
    get_query_execution_params,
    load_config,
    parse_args,
)
from gqe_bench.logger import configure_logging
from gqe_bench.suites.tpch import TpchSuite


class TestParseArgs:
    """Tests for parse_args."""

    def test_dataset_flag(self) -> None:
        args = parse_args(["--dataset", "/dataset"])
        assert isinstance(args.dataset, Path)
        assert args.dataset == Path("/dataset")

    def test_dataset_required(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([])

    def test_all_path_fields_convert(self) -> None:
        args = parse_args(
            [
                "--dataset",
                "/dataset",
                "--sql",
                "/q",
                "--solution",
                "/s",
                "--schema",
                "/sch",
                "--validate-dir",
                "/vd",
                "--output",
                "/o",
                "--server-bin",
                "/sb",
                "--cli-bin",
                "/cb",
                "--task-manager-bin",
                "/tb",
                "--swept-sqlite",
                "/sw",
            ]
        )
        path_fields = [f.name for f in FIELDS if f.type is Path]
        for name in path_fields:
            val = getattr(args, name)
            assert isinstance(val, Path), f"{name} is {type(val)}, expected Path"

    def test_json_config(self, tmp_path: Path) -> None:
        config = {"dataset": "/d", "sql": "/q", "solution": "/s"}
        config_file = tmp_path / "config.json5"
        config_file.write_text(json.dumps(config))
        args = parse_args(["--json", str(config_file)])
        assert isinstance(args.dataset, Path)
        assert isinstance(args.sql, Path)

    def test_unknown_field_rejected(self, tmp_path: Path) -> None:
        config = {"dataset": "/d", "bogus": True}
        config_file = tmp_path / "config.json5"
        config_file.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="Unknown fields"):
            parse_args(["--json", str(config_file)])


class TestConfigToArgs:
    """Tests for config_to_args."""

    def test_applies_defaults(self) -> None:
        args = config_to_args({"dataset": "/d"})
        assert args.repeat == 6
        assert args.suite_name == TpchSuite.NAME

    def test_overrides_defaults(self) -> None:
        args = config_to_args({"dataset": "/d", "repeat": 10})
        assert args.repeat == 10

    def test_query_overrides_stored(self) -> None:
        args = config_to_args(
            {
                "dataset": "/d",
                "query_overrides": [{"queries": ["1"], "num_partitions": [1]}],
            }
        )
        assert args.query_overrides == [{"queries": ["1"], "num_partitions": [1]}]

    def test_empty_query_overrides_when_absent(self) -> None:
        args = config_to_args({"dataset": "/d"})
        assert not hasattr(args, "query_overrides") or args.query_overrides is None


class TestLoadConfig:
    """Tests for load_config standalone."""

    def test_valid_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "valid.json5"
        config_file.write_text(json.dumps({"dataset": "/d", "repeat": 3}))
        config = load_config(str(config_file))
        assert config["dataset"] == "/d"
        assert config["repeat"] == 3


class TestGetQueryExecutionParams:
    """Tests for get_query_execution_params — override merging logic."""

    @staticmethod
    def _make_args(**overrides: object) -> Namespace:
        defaults = {
            "num_partitions": [1, 2, 4, 8],
            "num_workers": [1],
            "use_partition_pruning": [False],
            "use_overlap_mtx": [True],
            "read_use_zero_copy": [False],
            "filter_use_like_shift_and": [True],
            "join_use_hash_map_cache": [True],
            "join_use_unique_keys": [True],
            "join_use_perfect_hash": [False],
            "join_use_mark_join": [False],
            "aggregation_use_perfect_hash": [True],
            "use_ast_jit": [False],
            "query_source": ["sql"],
            "query_overrides": [],
        }
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_returns_all_query_fields(self) -> None:
        result = get_query_execution_params(self._make_args(), "1")
        for field in QUERY_FIELDS:
            assert field in result

    def test_uses_args_when_no_overrides(self) -> None:
        result = get_query_execution_params(
            self._make_args(num_partitions=[16, 32], num_workers=[2]), "1"
        )
        assert result["num_partitions"] == [16, 32]
        assert result["num_workers"] == [2]

    def test_applies_single_override(self) -> None:
        args = self._make_args(
            num_partitions=[1, 2, 4, 8],
            query_overrides=[{"queries": ["1", "2"], "num_partitions": [16]}],
        )
        assert get_query_execution_params(args, "1")["num_partitions"] == [16]

    def test_non_matching_query_uses_args(self) -> None:
        args = self._make_args(
            num_partitions=[1, 2],
            query_overrides=[{"queries": ["3"], "num_partitions": [16]}],
        )
        assert get_query_execution_params(args, "1")["num_partitions"] == [1, 2]

    def test_merges_multiple_overrides(self) -> None:
        args = self._make_args(
            num_partitions=[1],
            query_overrides=[
                {"queries": ["1", "2", "3"], "num_partitions": [2]},
                {"queries": ["1"], "num_partitions": [4]},
            ],
        )
        assert get_query_execution_params(args, "1")["num_partitions"] == [2, 4]

    def test_merge_deduplicates(self) -> None:
        args = self._make_args(
            num_partitions=[1, 2],
            query_overrides=[
                {"queries": ["1"], "num_partitions": [2, 3]},
                {"queries": ["1"], "num_partitions": [3, 4]},
            ],
        )
        assert get_query_execution_params(args, "1")["num_partitions"] == [2, 3, 4]

    def test_override_only_affects_specified_fields(self) -> None:
        args = self._make_args(
            num_partitions=[1, 2],
            num_workers=[1],
            query_overrides=[{"queries": ["1"], "num_partitions": [4]}],
        )
        result = get_query_execution_params(args, "1")
        assert result["num_partitions"] == [4]
        assert result["num_workers"] == [1]


class TestLogLevel:
    """Tests for the --log-level field and configure_logging."""

    def test_default_is_info(self) -> None:
        args = parse_args(["--dataset", "/d"])
        assert args.log_level == "INFO"

    def test_case_insensitive(self) -> None:
        args = parse_args(["--dataset", "/d", "--log-level", "debug"])
        assert args.log_level == "DEBUG"

    def test_quiet_accepted(self) -> None:
        args = parse_args(["--dataset", "/d", "--log-level", "quiet"])
        assert args.log_level == "QUIET"

    def test_invalid_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            parse_args(["--dataset", "/d", "--log-level", "bogus"])

    def test_critical_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            parse_args(["--dataset", "/d", "--log-level", "critical"])

    def test_json_config_honored(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json5"
        config_file.write_text(json.dumps({"dataset": "/d", "log_level": "warning"}))
        args = parse_args(["--json", str(config_file)])
        assert args.log_level == "WARNING"

    def test_configure_logging_sets_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}
        monkeypatch.setattr(logging, "basicConfig", lambda **kw: captured.update(kw))
        configure_logging(Namespace(log_level="DEBUG"))
        assert captured["level"] == logging.DEBUG

    def test_configure_logging_quiet_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}
        monkeypatch.setattr(logging, "disable", lambda level: captured.update(level=level))
        configure_logging(Namespace(log_level="QUIET"))
        assert captured["level"] == logging.CRITICAL


class TestDecimalTypeField:
    """`decimal_type` is a cross-table override field, like storage_device_kind:
    its column lives in data_info rather than a gqe_* table, so the sweep default
    has to come from _MANUAL_FIELDS. Without it the field defaults to None and the
    Cartesian product raises."""

    def test_omitted_key_still_yields_a_sweep_list(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json5"
        config_file.write_text("{}")
        args = parse_args(["--json", str(config_file)])
        assert args.decimal_type == ["double"]

    def test_config_value_overrides_the_default(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json5"
        config_file.write_text('{"decimal_type": ["double", "decimal"]}')
        args = parse_args(["--json", str(config_file)])
        assert args.decimal_type == ["double", "decimal"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
