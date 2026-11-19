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
import shutil
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
from gqe_bench.profiling import NsysError, NsysProfiler
from gqe_bench.suites.tpch import TpchSuite


class TestParseArgs:
    """Tests for parse_args."""

    def test_dataset_flag(self) -> None:
        args = parse_args(["--dataset", "/dataset"])
        assert isinstance(args.dataset, Path)
        assert args.dataset == Path("/dataset")

    def test_dataset_required(self) -> None:
        with pytest.raises(ValueError, match="dataset is required"):
            parse_args([])

    def test_dataset_required_in_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json5"
        config_file.write_text(json.dumps({"repeat": 3}))
        with pytest.raises(ValueError, match="dataset is required"):
            parse_args(["--json", str(config_file)])

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
        # Skip --nsys and the Path fields that require it: they are None unless
        # --nsys is passed. TestNsysArgs covers them.
        gates = {f.requires for f in FIELDS if f.requires is not None}
        path_fields = [
            f.name for f in FIELDS if f.type is Path and f.requires is None and f.name not in gates
        ]
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

    def test_sweep_values_typed_like_ddl(self) -> None:
        """CLI sweep values are converted to their column types, not left as strings."""
        args = parse_args(
            [
                "--dataset",
                "/d",
                "--num-workers",
                "1",
                "2",
                "--compression-ratio-threshold",
                "1.5",
                "--join-materialization-threshold",
                "0",
                "--zone-map-partition-size",
                "1000",
                "--use-cpu-compression",
                "false",
                "True",
                "--compression-format",
                "lz4",
            ]
        )
        assert args.num_workers == [1, 2]
        assert args.compression_ratio_threshold == [1.5]
        assert args.join_materialization_threshold == [0.0]
        assert all(type(v) is float for v in args.join_materialization_threshold)
        assert args.zone_map_partition_size == [1000]
        assert args.use_cpu_compression == [False, True]
        assert all(type(v) is bool for v in args.use_cpu_compression)
        assert args.compression_format == ["lz4"]

    def test_json_integers_become_floats_for_float_options(self, tmp_path: Path) -> None:
        """GQE rejects an integer literal for a double option, so JSON 0 is sent as 0.0."""
        config_file = tmp_path / "config.json5"
        config = {
            "dataset": "/d",
            "join_materialization_threshold": [0, 0.1],
            "query_overrides": [{"queries": ["1"], "join_materialization_threshold": [1]}],
        }
        config_file.write_text(json.dumps(config))
        args = parse_args(["--json", str(config_file)])
        assert args.join_materialization_threshold == [0.0, 0.1]
        assert all(type(v) is float for v in args.join_materialization_threshold)
        assert type(args.query_overrides[0]["join_materialization_threshold"][0]) is float

    def test_late_materialization_defaults_to_true(self) -> None:
        """The option matches GQE's default and accepts both values from the CLI."""
        assert parse_args(["--dataset", "/d"]).use_late_materialization == [True]
        args = parse_args(["--dataset", "/d", "--use-late-materialization", "false", "true"])
        assert args.use_late_materialization == [False, True]

    @pytest.mark.parametrize("value", ["1", "0", "yes", "maybe"])
    def test_invalid_bool_sweep_value_rejected(self, value: str) -> None:
        """Only true/false are accepted, as in the pre-gqe-cli runner."""
        with pytest.raises(SystemExit):
            parse_args(["--dataset", "/d", "--use-cpu-compression", value])


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
            "execution_backend": ["legacy_task_graph"],
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
            "use_late_materialization": [True],
            "join_materialization_threshold": [0.1],
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


class TestExecutionBackend:
    """Tests for execution-backend parsing and validation."""

    def test_default_is_legacy(self) -> None:
        args = parse_args(["--dataset", "/d"])
        assert args.execution_backend == ["legacy_task_graph"]

    def test_cli_accepts_both_backends(self) -> None:
        args = parse_args(
            [
                "--dataset",
                "/d",
                "--execution-backend",
                "legacy_task_graph",
                "qep_task_graph",
            ]
        )
        assert args.execution_backend == ["legacy_task_graph", "qep_task_graph"]

    def test_invalid_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid execution backend"):
            parse_args(["--dataset", "/d", "--execution-backend", "invalid"])

    def test_invalid_override_backend_rejected(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json5"
        config_file.write_text(
            json.dumps(
                {
                    "dataset": "/d",
                    "query_overrides": [{"queries": ["6"], "execution_backend": ["invalid"]}],
                }
            )
        )
        with pytest.raises(ValueError, match="Invalid execution backend"):
            parse_args(["--json", str(config_file)])

    @pytest.mark.parametrize("value", [None, ["legacy_task_graph"], {"name": "legacy_task_graph"}])
    def test_non_string_backend_rejected(self, tmp_path: Path, value: object) -> None:
        config_file = tmp_path / "config.json5"
        config_file.write_text(json.dumps({"dataset": "/d", "execution_backend": [value]}))
        with pytest.raises(ValueError, match="values must be strings"):
            parse_args(["--json", str(config_file)])


class TestNsysArgs:
    """`--nsys` resolution and the flags that depend on it.

    Unlike the *_bin fields, an unresolvable --nsys is fatal at parse time:
    nsys is an external tool, so no later Artifact.require() can rescue it.
    """

    @pytest.fixture(autouse=True)
    def _accept_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parsing --nsys runs nsys, and the paths these tests name do not exist."""
        monkeypatch.setattr(NsysProfiler, "verify_args", lambda *_: None)

    @pytest.mark.parametrize(
        ("given", "queried"),
        [(None, "nsys"), ("/opt/nsys/bin/nsys", "/opt/nsys/bin/nsys")],
    )
    def test_resolution_queries_which(
        self, monkeypatch: pytest.MonkeyPatch, given: str | None, queried: str
    ) -> None:
        """A bare --nsys resolves the const "nsys"; an explicit path resolves itself."""
        seen: list[str] = []

        def which(name: str) -> str:
            seen.append(name)
            return "/usr/local/bin/nsys"

        monkeypatch.setattr(shutil, "which", which)
        args = parse_args(["--dataset", "/d", "--nsys"] + ([given] if given else []))
        assert seen == [queried]
        assert args.nsys == Path("/usr/local/bin/nsys")

    def test_unresolvable_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(ValueError, match="nsys not found"):
            parse_args(["--dataset", "/d", "--nsys"])

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--nsys-trace", "x"),
            ("--nsys-launch-args", "x"),
            ("--nsys-args", "x"),
            ("--nsys-prefix", "x"),
            ("--nsys-output-dir", "x"),
            # Values equal to the --nsys defaults are rejected too.
            ("--nsys-trace", "cuda,nvtx"),
            ("--nsys-launch-args", ""),
            ("--nsys-args", ""),
            ("--nsys-prefix", ""),
        ],
    )
    def test_dependent_flags_require_nsys(self, flag: str, value: str) -> None:
        with pytest.raises(ValueError, match=f"{flag} requires --nsys"):
            parse_args(["--dataset", "/d", f"{flag}={value}"])

    def test_dependent_fields_none_without_nsys(self) -> None:
        args = parse_args(["--dataset", "/d"])
        assert args.nsys_trace is None
        assert args.nsys_launch_args is None
        assert args.nsys_args is None
        assert args.nsys_prefix is None
        assert args.nsys_output_dir is None

    def test_dependent_fields_default_with_nsys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nsys")
        args = parse_args(["--dataset", "/d", "--nsys"])
        assert args.nsys_trace == "cuda,nvtx"
        assert args.nsys_launch_args == ""
        assert args.nsys_args == ""
        assert args.nsys_prefix == ""
        assert args.nsys_output_dir == Path.cwd()

    @pytest.mark.parametrize(
        ("flag", "value", "match"),
        [
            # The attached "-o<path>" spelling is a bypass of a bare "-o" check.
            ("--nsys-args", "-o /tmp/mine", "owns the nsys output path"),
            ("--nsys-args", "--output=/tmp/mine", "owns the nsys output path"),
            ("--nsys-args", "-o/tmp/mine", "owns the nsys output path"),
            ("--nsys-args", "--session=mine", "owns the nsys session"),
            ("--nsys-launch-args", "-t cuda", "use --nsys-trace"),
            ("--nsys-launch-args", "--trace=cuda", "use --nsys-trace"),
            ("--nsys-launch-args", "--session=mine", "owns the nsys session"),
        ],
    )
    def test_reserved_flags_rejected(
        self, monkeypatch: pytest.MonkeyPatch, flag: str, value: str, match: str
    ) -> None:
        # The equals form is required for any value starting with '-'; argparse
        # otherwise reads it as a flag of its own.
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nsys")
        with pytest.raises(ValueError, match=match):
            parse_args(["--dataset", "/d", "--nsys", f"{flag}={value}"])

    def test_json_config_honored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nsys")
        config_file = tmp_path / "config.json5"
        config_file.write_text(json.dumps({"dataset": "/d", "nsys": "nsys", "nsys_prefix": "sf1k"}))
        args = parse_args(["--json", str(config_file)])
        assert args.nsys == Path("/usr/bin/nsys")
        assert args.nsys_prefix == "sf1k"

    def test_args_nsys_rejects_are_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nsys")

        def reject(*_: object) -> None:
            raise NsysError(["nsys", "profile"], 1, "unrecognised option '--bogus'")

        monkeypatch.setattr(NsysProfiler, "verify_args", reject)
        with pytest.raises(ValueError, match="unrecognised option"):
            parse_args(["--dataset", "/d", "--nsys", "--nsys-args=--bogus"])

    def test_verify_args_receives_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nsys")
        seen: list[tuple[object, ...]] = []
        monkeypatch.setattr(NsysProfiler, "verify_args", lambda *a: seen.append(a))
        parse_args(["--dataset", "/d", "--nsys"])
        assert seen == [(Path("/usr/bin/nsys"), "cuda,nvtx", "", "")]


class TestDecimalTypeField:
    """`decimal_type` is a cross-table override field, like storage_device_kind:
    its column lives in data_info rather than a gqe_* table, so the sweep default
    has to come from _MANUAL_FIELDS. Without it the field defaults to None and the
    Cartesian product raises."""

    def test_omitted_key_still_yields_a_sweep_list(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json5"
        config_file.write_text('{"dataset": "/d"}')
        args = parse_args(["--json", str(config_file)])
        assert args.decimal_type == ["double"]

    def test_config_value_overrides_the_default(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json5"
        config_file.write_text('{"dataset": "/d", "decimal_type": ["double", "decimal"]}')
        args = parse_args(["--json", str(config_file)])
        assert args.decimal_type == ["double", "decimal"]


class TestSerializeValidation:
    """Tests to ensure that serialize rejects incompatible flags."""

    def test_both_flags_rejected(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            parse_args(["--dataset", "/d", "--serialize-to-disk", "--deserialize-from-disk"])

    def test_serialize_requires_load_all_data(self) -> None:
        with pytest.raises(ValueError, match="requires --load-all-data"):
            parse_args(["--dataset", "/d", "--serialize-to-disk"])

    def test_serialize_rejects_secondary_compression(self) -> None:
        with pytest.raises(ValueError, match="secondary_compression_format"):
            parse_args(
                [
                    "--dataset",
                    "/d",
                    "--serialize-to-disk",
                    "--load-all-data",
                    "--secondary-compression-format",
                    "ans",
                ]
            )

    def test_serialize_rejects_a_sweep_that_includes_secondary_compression(self) -> None:
        with pytest.raises(ValueError, match="secondary_compression_format"):
            parse_args(
                [
                    "--dataset",
                    "/d",
                    "--serialize-to-disk",
                    "--load-all-data",
                    "--secondary-compression-format",
                    "none",
                    "ans",
                ]
            )

    def test_serialize_accepts_none(self) -> None:
        args = parse_args(["--dataset", "/d", "--serialize-to-disk", "--load-all-data"])
        assert args.serialize_to_disk is True
        assert args.secondary_compression_format == ["none"]

    def test_deserialize_allows_secondary_compression(self) -> None:
        args = parse_args(
            [
                "--dataset",
                "/d",
                "--deserialize-from-disk",
                "--secondary-compression-format",
                "ans",
            ]
        )
        assert args.deserialize_from_disk is True
        assert args.secondary_compression_format == ["ans"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
