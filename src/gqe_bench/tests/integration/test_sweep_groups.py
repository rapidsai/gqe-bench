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
Tests for sweep group generation over handcoded queries.

Query discovery serializes every handcoded plan, so these need the physical-plan
protos and therefore a GQE_FETCH_ENGINE=ON build. The rest of the sweep-group
tests are source-agnostic and live in tests/test_sweep_groups.py.

Usage:
    pytest gqe_bench/tests/integration/test_sweep_groups.py -v
"""

from argparse import Namespace
from pathlib import Path

import pytest

from gqe_bench.arguments import parse_args
from gqe_bench.gqe_params import generate_sweep_groups
from gqe_bench.schema import DataLoadGroup


def _groups(args: Namespace, schema_ddl: str) -> list[DataLoadGroup]:
    return generate_sweep_groups(args, schema_ddl)


class TestHandcodedSweepGroups:
    """Tests for generate_sweep_groups on the handcoded source."""

    def test_variants_group_with_base_query(self, fake_dataset: Path, fake_schema_ddl: str) -> None:
        args = parse_args(["--query-source", "handcoded", "--dataset", str(fake_dataset)])
        args.load_all_data = False  # per-group loads, so grouping is by base query
        names_per_group = [{q.name for q, _ in g.queries} for g in _groups(args, fake_schema_ddl)]
        group_with_2 = next(names for names in names_per_group if "2" in names)
        assert "2_fused_filter" in group_with_2  # variant co-bucketed with base query 2
        group_with_3 = next(names for names in names_per_group if "3" in names)
        assert "2" not in group_with_3  # a different base query is in a different group


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
