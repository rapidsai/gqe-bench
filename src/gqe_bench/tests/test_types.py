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
Tests for gqe_bench type definitions.

Usage:
    pytest gqe_bench/tests/test_types.py -v
"""

from dataclasses import FrozenInstanceError, fields

import pytest

from gqe_bench.query_source import QuerySource
from gqe_bench.schema import DATA_INFO_MAPPING, QUERY_PARAMS_MAPPING, DataInfo, QueryParams
from gqe_bench.tests.helpers import make_data_info


class TestQuerySource:
    """Tests for the QuerySource enum and from_db conversion."""

    def test_from_db_sql(self) -> None:
        assert QuerySource.from_db("sql") == QuerySource.SQL

    def test_from_db_handcoded(self) -> None:
        assert QuerySource.from_db("handcoded") == QuerySource.HANDCODED

    def test_from_db_substrait_legacy(self) -> None:
        """Legacy 'substrait' values in existing DBs map to SQL."""
        assert QuerySource.from_db("substrait") == QuerySource.SQL

    def test_from_db_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            QuerySource.from_db("bogus")


class TestDataInfo:
    """Tests for DataInfo — frozen, hashable (used as dict key in plan grouping)."""

    def test_frozen(self) -> None:
        di = make_data_info()
        with pytest.raises(FrozenInstanceError):
            di.location = "/y"

    def test_hashable(self) -> None:
        """DataInfo is used as dict key in plans_by_data grouping."""
        di = make_data_info()
        {di: True}

    def test_immutability(self) -> None:
        """Two identical DataInfos are equal (used for dedup in sweep grouping)."""
        di1 = make_data_info()
        di2 = make_data_info()
        assert di1 == di2
        assert hash(di1) == hash(di2)


class TestSweepFieldsConsistency:
    """Verify schema-derived sweep fields match dataclass fields."""

    def test_data_info_sweep_fields(self) -> None:
        field_names = {f.name for f in fields(DataInfo)}
        for field_name in DATA_INFO_MAPPING.sweep_fields:
            assert field_name in field_names, f"DataInfo missing field: {field_name}"

    def test_query_params_sweep_fields(self) -> None:
        field_names = {f.name for f in fields(QueryParams)}
        for field_name in QUERY_PARAMS_MAPPING.sweep_fields:
            assert field_name in field_names, f"QueryParams missing field: {field_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
