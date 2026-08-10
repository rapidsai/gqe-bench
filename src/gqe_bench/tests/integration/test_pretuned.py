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
Tests for pretuned group generation.

Usage:
    pytest gqe_bench/tests/test_pretuned.py -v --database /path/to/tpch.db3
"""

from pathlib import Path

import pytest

from gqe_bench.arguments import parse_args
from gqe_bench.gqe_params import generate_pretuned_groups
from gqe_bench.query_source import QuerySource


class TestPretunedPlans:
    """Tests against a real .db3 from benchmark sweeps."""

    def test_generates_plans(self, db_path: Path) -> None:
        args = parse_args(["--swept-sqlite", str(db_path), "--dataset", "/tmp/ds"])
        groups = generate_pretuned_groups(args)
        assert len(groups) > 0
        assert sum(len(g.queries) for g in groups) > 0

    def test_with_solution(self, db_path: Path, tmp_path: Path) -> None:
        sol = tmp_path / "solutions"
        sol.mkdir()
        for i in range(1, 23):
            (sol / f"q{i}.parquet").touch()
        args = parse_args(
            [
                "--swept-sqlite",
                str(db_path),
                "--solution",
                str(sol),
                "--dataset",
                "/tmp/ds",
            ]
        )
        groups = generate_pretuned_groups(args)
        assert any(query.reference_file is not None for g in groups for query, _ in g.queries)

    def test_without_solution(self, db_path: Path) -> None:
        args = parse_args(["--swept-sqlite", str(db_path), "--dataset", "/tmp/ds"])
        groups = generate_pretuned_groups(args)
        assert all(query.reference_file is None for g in groups for query, _ in g.queries)

    def test_query_source_types(self, db_path: Path) -> None:
        args = parse_args(["--swept-sqlite", str(db_path), "--dataset", "/tmp/ds"])
        for g in generate_pretuned_groups(args):
            for query, _qp in g.queries:
                assert isinstance(query.source, QuerySource)

    def test_reference_file_types(self, db_path: Path) -> None:
        args = parse_args(["--swept-sqlite", str(db_path), "--dataset", "/tmp/ds"])
        for g in generate_pretuned_groups(args):
            for query, _qp in g.queries:
                if query.reference_file is not None:
                    assert isinstance(query.reference_file, Path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
