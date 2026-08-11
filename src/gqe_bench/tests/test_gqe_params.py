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
Tests for parameter generation internals.

Usage:
    pytest gqe_bench/tests/test_gqe_params.py -v
"""

from pathlib import Path

import pytest

from gqe_bench.arguments import config_to_args
from gqe_bench.gqe_params import (
    _prune,
    _warn_field_dependencies,
)
from gqe_bench.suites.tpch import TpchSuite
from gqe_bench.tests.helpers import make_data_info, make_query, make_query_params


class TestInferScaleFactor:
    def test_sf100(self) -> None:
        assert TpchSuite.infer_scale_factor(Path("/data/sf100_dataset")) == 100.0

    def test_sf1000(self) -> None:
        assert TpchSuite.infer_scale_factor(Path("/data/sf1000_chunk16m")) == 1000.0

    def test_sf1(self) -> None:
        assert TpchSuite.infer_scale_factor(Path("/data/sf1_id64")) == 1.0

    def test_fractional_sf(self) -> None:
        assert TpchSuite.infer_scale_factor(Path("/data/sf0.01_id64")) == 0.01

    def test_k_suffix(self) -> None:
        assert TpchSuite.infer_scale_factor(Path("/data/sf1k_chunk16m")) == 1000.0
        assert TpchSuite.infer_scale_factor(Path("/data/sf10k_id64")) == 10000.0

    def test_returns_float_type(self) -> None:
        result = TpchSuite.infer_scale_factor(Path("/data/sf100_dataset"))
        assert isinstance(result, float)

    def test_resolves_symlink(self, tmp_path: Path) -> None:
        real = tmp_path / "sf0.01_data"
        real.mkdir()
        link = tmp_path / "sf001"
        link.symlink_to(real)
        # Input path has misleading name "sf001"; real path name is "sf0.01".
        # Resolver should follow the symlink and produce the correct SF.
        assert TpchSuite.infer_scale_factor(link) == 0.01

    def test_no_match(self) -> None:
        assert TpchSuite.infer_scale_factor(Path("/no/scale/here")) == 1.0


class TestQuerySqlScaleFactor:
    def test_q11_default_threshold_unchanged(self) -> None:
        sql = TpchSuite.query_sql("11")
        assert "0.0001000000" in sql

    def test_q11_sf1_same_as_default(self) -> None:
        assert TpchSuite.query_sql("11", 1.0) == TpchSuite.query_sql("11")

    def test_q11_fractional_sf_substitutes_threshold(self) -> None:
        sql = TpchSuite.query_sql("11", 0.01)
        assert "0.0001000000" not in sql
        # 0.0001 / 0.01 = 0.01
        assert "0.0100000000" in sql

    def test_q11_large_sf_substitutes_threshold(self) -> None:
        sql = TpchSuite.query_sql("11", 100.0)
        assert "0.0001000000" not in sql
        # 0.0001 / 100 = 0.000001
        assert "0.0000010000" in sql

    def test_non_q11_unchanged_by_scale_factor(self) -> None:
        assert TpchSuite.query_sql("1", 100.0) == TpchSuite.query_sql("1", 1.0)
        assert TpchSuite.query_sql("15", 0.01) == TpchSuite.query_sql("15")

    def test_zero_sf_raises(self) -> None:
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            TpchSuite.query_sql("11", 0.0)

    def test_negative_sf_raises(self) -> None:
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            TpchSuite.query_sql("11", -1.0)


class TestWarnFieldDependencies:
    def test_compression_level_without_cpu_compression(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        args = config_to_args(
            {"dataset": "/tmp/ds", "compression_level": [5], "use_cpu_compression": [False]}
        )
        _warn_field_dependencies(args)
        assert "compression_level" in caplog.text

    def test_compression_level_with_cpu_compression(self, caplog: pytest.LogCaptureFixture) -> None:
        args = config_to_args(
            {"dataset": "/tmp/ds", "compression_level": [5], "use_cpu_compression": [True]}
        )
        _warn_field_dependencies(args)
        assert "compression_level" not in caplog.text

    def test_zone_map_without_partition_pruning(self, caplog: pytest.LogCaptureFixture) -> None:
        args = config_to_args(
            {
                "dataset": "/tmp/ds",
                "zone_map_partition_size": [200000],
                "use_partition_pruning": [False],
            }
        )
        _warn_field_dependencies(args)
        assert "zone_map_partition_size" in caplog.text

    def test_partition_pruning_without_zone_map(self, caplog: pytest.LogCaptureFixture) -> None:
        args = config_to_args(
            {
                "dataset": "/tmp/ds",
                "zone_map_partition_size": None,
                "use_partition_pruning": [True],
            }
        )
        _warn_field_dependencies(args)
        assert "zone_map_partition_size" in caplog.text

    def test_both_set_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        args = config_to_args(
            {
                "dataset": "/tmp/ds",
                "zone_map_partition_size": [200000],
                "use_partition_pruning": [True],
                "compression_level": [10],
                "use_cpu_compression": [False],
            }
        )
        _warn_field_dependencies(args)
        assert "zone_map_partition_size" not in caplog.text
        assert "compression_level" not in caplog.text


class TestPrune:
    """Each pruning rule tested in isolation."""

    def test_valid_combo_not_pruned(self) -> None:
        assert not _prune(make_query(), make_query_params(), make_data_info())

    def test_zero_copy_partitions_mismatch(self) -> None:
        assert _prune(
            make_query(),
            make_query_params(read_use_zero_copy=True, num_partitions=4),
            make_data_info(num_row_groups=8),
        )

    def test_zero_copy_parquet_file(self) -> None:
        assert _prune(
            make_query(),
            make_query_params(read_use_zero_copy=True, num_partitions=8),
            make_data_info(num_row_groups=8, storage_device_kind="parquet_file"),
        )

    def test_zero_copy_with_compression(self) -> None:
        assert _prune(
            make_query(),
            make_query_params(read_use_zero_copy=True, num_partitions=8),
            make_data_info(num_row_groups=8, compression_format="snappy"),
        )

    def test_zero_copy_valid(self) -> None:
        assert not _prune(
            make_query(),
            make_query_params(read_use_zero_copy=True, num_partitions=8),
            make_data_info(num_row_groups=8),
        )

    def test_workers_gt_partitions(self) -> None:
        assert _prune(
            make_query(),
            make_query_params(num_workers=8, num_partitions=4),
            make_data_info(),
        )

    def test_compression_with_overlap_mtx(self) -> None:
        assert _prune(
            make_query(),
            make_query_params(use_overlap_mtx=True),
            make_data_info(compression_format="snappy"),
        )

    def test_perfect_hash_equals_hash_map_cache(self) -> None:
        assert _prune(
            make_query(),
            make_query_params(join_use_perfect_hash=True, join_use_hash_map_cache=True),
            make_data_info(),
        )
        assert _prune(
            make_query(),
            make_query_params(join_use_perfect_hash=False, join_use_hash_map_cache=False),
            make_data_info(),
        )

    def test_large_sf_small_partitions_pruned(self) -> None:
        """Lowercase query names match the case-sensitive regex."""
        assert _prune(
            make_query(name="q1"),
            make_query_params(num_partitions=2),
            make_data_info(scale_factor=1000),
        )

    def test_large_sf_small_partitions_uppercase_not_pruned(self) -> None:
        """Uppercase 'Q1' does not match the case-sensitive regex."""
        assert not _prune(
            make_query(name="Q1"),
            make_query_params(num_partitions=2),
            make_data_info(scale_factor=1000),
        )

    def test_large_sf_exempt_q11(self) -> None:
        assert not _prune(
            make_query(name="q11"),
            make_query_params(num_partitions=2),
            make_data_info(scale_factor=1000),
        )

    def test_large_sf_exempt_q20(self) -> None:
        assert not _prune(
            make_query(name="q20"),
            make_query_params(num_partitions=2),
            make_data_info(scale_factor=1000),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
