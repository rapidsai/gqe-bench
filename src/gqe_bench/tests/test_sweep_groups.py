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
Tests for sweep group generation.

Usage:
    pytest gqe_bench/tests/test_sweep_groups.py -v
"""

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

import gqe_bench.gqe_params as gqe_params
from gqe_bench.arguments import parse_args
from gqe_bench.gqe_params import generate_pretuned_groups, generate_sweep_groups
from gqe_bench.schema import NUM_GPUS, Q_NAME, Q_SOURCE, SCALE_FACTOR, DataLoadGroup
from gqe_bench.suites.tpch import TpchSuite


def _groups(args: Namespace, schema_ddl: str) -> list[DataLoadGroup]:
    return generate_sweep_groups(args, schema_ddl)


class TestSweepGroups:
    """Tests for generate_sweep_groups."""

    def test_payload_cache_preserves_variants_and_is_local_to_generation(
        self, fake_dataset: Path, fake_schema_ddl: str
    ) -> None:
        args = parse_args(
            ["--dataset", str(fake_dataset), "--queries", "1", "--query-source", "handcoded"]
        )
        args.num_workers = [1, 2]
        args.num_partitions = [4]
        args.num_row_groups = [4, 8]
        args.decimal_type = ["double", "decimal"]
        args.execution_backend = ["legacy_task_graph", "qep_task_graph"]
        args.join_use_perfect_hash = [False]
        args.join_use_hash_map_cache = [True]
        args.aggregation_use_perfect_hash = [False]

        # Distinct payloads catch cached content assigned to the wrong variant.
        with patch.object(
            TpchSuite,
            "resolve_content",
            side_effect=lambda *inputs: f"{inputs[-2]}:{inputs[-1]}".encode("utf-8"),
        ) as resolve:
            groups = _groups(args, fake_schema_ddl)
            assert sum(len(g.queries) for g in groups) == 16
            assert resolve.call_count == 4
            assert {call.args[-2:] for call in resolve.call_args_list} == {
                (decimal, backend)
                for decimal in args.decimal_type
                for backend in args.execution_backend
            }
            for group in groups:
                for query, params in group.queries:
                    expected = (
                        f"{group.data_info.decimal_type}:{params.execution_backend}"
                    ).encode("utf-8")
                    assert query.content == expected

            assert _groups(args, fake_schema_ddl) == groups
            assert resolve.call_count == 8

    def test_tpch_autogen(self, fake_dataset: Path, fake_schema_ddl: str) -> None:
        args = parse_args(["--dataset", str(fake_dataset)])
        groups = _groups(args, fake_schema_ddl)
        assert len(groups) > 0
        assert sum(len(g.queries) for g in groups) > 0

    def test_explicit_sql_dir(
        self, tmp_path: Path, fake_dataset: Path, fake_schema_ddl: str
    ) -> None:
        sql_dir = tmp_path / "sql"
        sql_dir.mkdir()
        for i in range(1, 23):
            (sql_dir / f"q{i}.sql").write_text(f"SELECT {i};")
        args = parse_args(["--sql", str(sql_dir), "--dataset", str(fake_dataset)])
        groups = _groups(args, fake_schema_ddl)
        assert sum(len(g.queries) for g in groups) > 0

    def test_handcoded_unknown_suite_raises(self, fake_dataset: Path, fake_schema_ddl: str) -> None:
        args = parse_args(
            [
                "--query-source",
                "sql",
                "--suite-name",
                "TPC-DS",
                "--dataset",
                str(fake_dataset),
            ]
        )
        with pytest.raises(ValueError, match="does not support"):
            _groups(args, fake_schema_ddl)

    def test_workers_le_partitions(self, fake_dataset: Path, fake_schema_ddl: str) -> None:
        args = parse_args(["--dataset", str(fake_dataset)])
        for g in _groups(args, fake_schema_ddl):
            for _query, qp in g.queries:
                assert qp.num_workers <= qp.num_partitions

    def test_zero_copy_requires_no_compression(
        self, fake_dataset: Path, fake_schema_ddl: str
    ) -> None:
        args = parse_args(["--dataset", str(fake_dataset)])
        for g in _groups(args, fake_schema_ddl):
            for _query, qp in g.queries:
                if qp.read_use_zero_copy:
                    assert g.data_info.compression_format == "none"

    def test_perfect_hash_ne_hash_map_cache(self, fake_dataset: Path, fake_schema_ddl: str) -> None:
        args = parse_args(["--dataset", str(fake_dataset)])
        for g in _groups(args, fake_schema_ddl):
            for _query, qp in g.queries:
                assert qp.join_use_perfect_hash != qp.join_use_hash_map_cache

    def test_qep_pruned_for_multiple_gpus(
        self,
        fake_dataset: Path,
        fake_schema_ddl: str,
    ) -> None:
        """Prune qep_task_graph and warn when it is the only requested backend."""
        args = parse_args(
            [
                "--dataset",
                str(fake_dataset),
                "--queries",
                "1",
                "--num-gpus",
                "2",
                "--execution-backend",
                "qep_task_graph",
            ]
        )
        assert _groups(args, fake_schema_ddl) == []

    def test_mixed_backends_pruned_for_multiple_gpus(
        self,
        fake_dataset: Path,
        fake_schema_ddl: str,
    ) -> None:
        """Prune qep_task_graph but retain legacy combinations in a mixed-backend sweep."""
        args = parse_args(
            [
                "--dataset",
                str(fake_dataset),
                "--queries",
                "1",
                "--num-gpus",
                "2",
                "--execution-backend",
                "legacy_task_graph",
                "qep_task_graph",
            ]
        )
        groups = _groups(args, fake_schema_ddl)
        assert groups
        assert all(
            qp.execution_backend == "legacy_task_graph"
            for group in groups
            for _query, qp in group.queries
        )

    def test_pretuned_uses_recorded_num_gpus(
        self,
        tmp_path: Path,
        fake_dataset: Path,
        fake_schema_ddl: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configure pretuned execution with the recorded GPU count."""
        sql_dir = tmp_path / "sql"
        sql_dir.mkdir()
        (sql_dir / "q1.sql").write_text("SELECT 1;")
        args = parse_args(
            [
                "--swept-sqlite",
                str(tmp_path / "sweep.db3"),
                "--dataset",
                str(fake_dataset),
                "--sql",
                str(sql_dir),
                "--queries",
                "1",
            ]
        )

        def query_best_parameters(_paths: list[Path]) -> list[dict]:
            return [
                {
                    NUM_GPUS: 2,
                    Q_NAME: "1",
                    Q_SOURCE: "sql",
                    SCALE_FACTOR: 100.0,
                    "p_execution_backend": "legacy_task_graph",
                }
            ]

        monkeypatch.setattr(gqe_params, "query_best_parameters", query_best_parameters)

        assert generate_pretuned_groups(args, fake_schema_ddl)
        assert args.num_gpus == 2

    def test_pretuned_restores_boolean_flags(
        self,
        tmp_path: Path,
        fake_dataset: Path,
        fake_schema_ddl: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Convert SQLite INTEGER 0/1 flags back to bool.

        The engine rejects ``SET <flag> TO 1`` for boolean options.
        """
        sql_dir = tmp_path / "sql"
        sql_dir.mkdir()
        (sql_dir / "q1.sql").write_text("SELECT 1;")
        args = parse_args(
            [
                "--swept-sqlite",
                str(tmp_path / "sweep.db3"),
                "--dataset",
                str(fake_dataset),
                "--sql",
                str(sql_dir),
                "--queries",
                "1",
            ]
        )

        def query_best_parameters(_paths: list[Path]) -> list[dict]:
            return [
                {
                    NUM_GPUS: 1,
                    Q_NAME: "1",
                    Q_SOURCE: "sql",
                    SCALE_FACTOR: 100.0,
                    "p_execution_backend": "legacy_task_graph",
                    "p_num_workers": 4,
                    "p_join_use_perfect_hash": 1,
                    "p_use_overlap_mtx": 0,
                    "de_use_cpu_compression": 1,
                    "de_compression_level": 12,
                }
            ]

        monkeypatch.setattr(gqe_params, "query_best_parameters", query_best_parameters)

        groups = generate_pretuned_groups(args, fake_schema_ddl)
        assert len(groups) == 1
        data_info = groups[0].data_info
        [(_query, qp)] = groups[0].queries

        assert data_info.use_cpu_compression is True
        assert qp.join_use_perfect_hash is True
        assert qp.use_overlap_mtx is False
        # Non-boolean INTEGER fields keep their integer values.
        assert data_info.compression_level == 12
        assert qp.num_workers == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("json_config", [False, True])
def test_shmem_policy_creates_distinct_groups(
    fake_dataset: Path, fake_schema_ddl: str, tmp_path: Path, json_config: bool
) -> None:
    """Policy is a data dimension and accepts names without a Python allowlist."""
    policies = ["single_arena_unplaced", "future_engine_policy"]
    if json_config:
        import json

        config = tmp_path / "policies.json5"
        config.write_text(
            json.dumps({"dataset": str(fake_dataset), "queries": ["1"], "shmem_policy": policies})
        )
        args = parse_args(["--json", str(config)])
    else:
        args = parse_args(
            ["--dataset", str(fake_dataset), "--queries", "1", "--shmem-policy", *policies]
        )
    groups = _groups(args, fake_schema_ddl)
    assert {g.data_info.shmem_policy for g in groups} == set(policies)
