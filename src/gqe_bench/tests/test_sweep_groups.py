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

import pytest

from gqe_bench.arguments import parse_args
from gqe_bench.gqe_params import generate_sweep_groups
from gqe_bench.schema import DataLoadGroup


def _groups(args: Namespace, schema_ddl: str) -> list[DataLoadGroup]:
    return generate_sweep_groups(args, schema_ddl)


class TestSweepGroups:
    """Tests for generate_sweep_groups."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
