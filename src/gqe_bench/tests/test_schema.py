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

"""Tests for benchmark schema views and best-parameter queries."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import gqe_bench.schema as schema
from gqe_bench.schema import NUM_GPUS, Q_NAME

_GQE_SUT_ID = 2717457836325482278


def _create_params_db(
    path: Path,
    rows: list[tuple[int, str, float, int]],
    *,
    view_has_num_gpus: bool,
    gpu_counts: dict[int, int],
) -> None:
    """Create the minimal database shape consumed by parameter selection."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE best_rows("
            "e_id INTEGER, q_name TEXT, r_avg_duration_s REAL, num_gpus INTEGER, "
            "d_decimal_type TEXT)"
        )
        conn.executemany("INSERT INTO best_rows VALUES (?, ?, ?, ?, 'double')", rows)
        selected_columns = "e_id, q_name, r_avg_duration_s, d_decimal_type"
        if view_has_num_gpus:
            selected_columns += ", num_gpus"
        conn.execute(
            f"CREATE VIEW {schema.VIEW_BEST_PARAMS_VALIDATED} "
            f"AS SELECT {selected_columns} FROM best_rows"
        )

        conn.execute(
            "CREATE TABLE experiment_gpu("
            "eg_experiment_id INTEGER, eg_gpu_info_id INTEGER, eg_cuda_index INTEGER)"
        )
        gpu_rows = [
            (experiment_id, cuda_index, cuda_index)
            for experiment_id, count in gpu_counts.items()
            for cuda_index in range(count)
        ]
        conn.executemany("INSERT INTO experiment_gpu VALUES (?, ?, ?)", gpu_rows)


def test_validated_view_counts_experiment_gpus() -> None:
    """The current schema records GPU cardinality in the pretuned source view."""
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(schema._load_base_schema())
        conn.executescript(schema._load_gqe_extension())
        conn.execute("INSERT INTO data_info(d_id) VALUES (1)")
        conn.execute("INSERT INTO query_info(q_id, q_name) VALUES (1, 'Q1')")
        conn.execute(
            "INSERT INTO gqe_parameters(p_id, p_sut_info_id) VALUES (1, 2717457836325482278)"
        )
        conn.execute("INSERT INTO gqe_data_info_ext(de_id, de_data_info_id) VALUES (1, 1)")
        conn.execute(
            "INSERT INTO experiment("
            "e_id, e_sut_info_id, e_parameters_id, e_data_info_id, e_query_info_id, "
            "e_sample_size, e_data_info_ext_id) "
            "VALUES (1, 2717457836325482278, 1, 1, 1, 2, 1)"
        )
        conn.executemany(
            "INSERT INTO run(r_experiment_id, r_number, r_duration_s) VALUES (1, ?, ?)",
            [(0, 1.0), (1, 1.0)],
        )
        conn.executemany(
            "INSERT INTO experiment_gpu VALUES (1, ?, ?)",
            [(1, 0), (2, 1)],
        )

        row = conn.execute(f"SELECT {NUM_GPUS} FROM {schema.VIEW_BEST_PARAMS_VALIDATED}").fetchone()

    assert row == (2,)


def test_legacy_validated_view_uses_experiment_gpu(tmp_path: Path) -> None:
    """Views created before num_gpus derive it from the junction table."""
    db_path = tmp_path / "legacy.db3"
    _create_params_db(
        db_path,
        [(1, "Q1", 1.0, 2), (2, "Q2", 2.0, 1)],
        view_has_num_gpus=False,
        gpu_counts={1: 2, 2: 2},
    )

    rows = schema._query_validated_params(db_path)

    assert [row[Q_NAME] for row in rows] == ["1", "2"]
    assert {row[NUM_GPUS] for row in rows} == {2}


def test_mixed_gpu_counts_rejected(tmp_path: Path) -> None:
    """One pretuned run cannot use results recorded with different GPU counts."""
    two_gpu_db = tmp_path / "two_gpu.db3"
    one_gpu_db = tmp_path / "one_gpu.db3"
    _create_params_db(
        two_gpu_db,
        [(1, "Q1", 1.0, 2)],
        view_has_num_gpus=True,
        gpu_counts={1: 2},
    )
    _create_params_db(
        one_gpu_db,
        [(1, "Q1", 2.0, 1)],
        view_has_num_gpus=True,
        gpu_counts={1: 1},
    )

    with pytest.raises(ValueError, match="multiple GPU counts: 1, 2"):
        schema.query_best_parameters([two_gpu_db, one_gpu_db])


def _create_benchmark_db(path: Path, measurements: dict[str, list[tuple[float, bool]]]) -> None:
    """Create a real-schema database with one measured trial per experiment."""
    with sqlite3.connect(path) as conn:
        conn.executescript(schema._load_base_schema())
        conn.executescript(schema._load_gqe_extension())
        conn.execute(
            "INSERT INTO query_info (q_id, q_name, q_suite, q_source) "
            "VALUES (1, 'Q1', 'TPC-H', 'sql')"
        )

        experiment_id = 0
        for data_id, (decimal_type, type_measurements) in enumerate(measurements.items(), start=1):
            conn.execute(
                "INSERT INTO data_info (d_id, d_decimal_type, d_scale_factor) " "VALUES (?, ?, 1)",
                (data_id, decimal_type),
            )
            conn.execute(
                "INSERT INTO gqe_data_info_ext (de_id, de_data_info_id) VALUES (?, ?)",
                (data_id, data_id),
            )

            for duration, validated in type_measurements:
                experiment_id += 1
                conn.execute(
                    "INSERT INTO gqe_parameters "
                    "(p_id, p_sut_info_id, p_num_partitions) VALUES (?, ?, ?)",
                    (experiment_id, _GQE_SUT_ID, experiment_id),
                )
                conn.execute(
                    "INSERT INTO experiment "
                    "(e_id, e_sut_info_id, e_parameters_id, e_data_info_id, "
                    "e_query_info_id, e_sample_size, e_data_info_ext_id) "
                    "VALUES (?, ?, ?, ?, 1, ?, ?)",
                    (
                        experiment_id,
                        _GQE_SUT_ID,
                        experiment_id,
                        data_id,
                        2 if validated else 3,
                        data_id,
                    ),
                )
                conn.executemany(
                    "INSERT INTO run (r_experiment_id, r_number, r_duration_s) " "VALUES (?, ?, ?)",
                    ((experiment_id, 0, duration + 100), (experiment_id, 1, duration)),
                )


def test_best_parameter_views_group_by_decimal_type(tmp_path: Path) -> None:
    db_path = tmp_path / "sweep.db3"
    _create_benchmark_db(
        db_path,
        {
            "double": [(1.0, True), (5.0, True)],
            "decimal": [(5.0, True), (10.0, True), (5.0, False)],
        },
    )

    with sqlite3.connect(db_path) as conn:
        best = conn.execute(
            "SELECT e_id, d_decimal_type, r_avg_duration_s FROM gqe_best_parameters"
        ).fetchall()
        validated = conn.execute(
            "SELECT e_id, d_decimal_type, r_avg_duration_s " "FROM gqe_best_parameters_validated"
        ).fetchall()

    # The double experiment with duration 5 must not match decimal's minimum.
    # Both tied decimal experiments are best, but only the fully validated one
    # belongs in the validated view.
    assert set(best) == {(1, "double", 1.0), (3, "decimal", 5.0), (5, "decimal", 5.0)}
    assert set(validated) == {(1, "double", 1.0), (3, "decimal", 5.0)}


def test_query_best_parameters_preserves_decimal_types_across_databases(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.db3"
    second = tmp_path / "second.db3"
    _create_benchmark_db(first, {"double": [(1.0, True)], "decimal": [(5.0, True)]})
    _create_benchmark_db(second, {"double": [(2.0, True)], "decimal": [(4.0, True)]})

    rows = schema.query_best_parameters([first, second])
    by_type = {row["d_decimal_type"]: row for row in rows}

    assert set(by_type) == {"double", "decimal"}
    assert by_type["double"]["r_avg_duration_s"] == 1.0
    assert by_type["decimal"]["r_avg_duration_s"] == 4.0


@pytest.mark.parametrize("storage_kind", ["shared_segment_memory", "parquet_file"])
def test_shmem_policy_recording_and_best_view(tmp_path: Path, storage_kind: str) -> None:
    """Policy distinguishes extension rows and survives selection for replay."""
    from gqe_bench.recording import _GqeConnection, insert_data_dimensions
    from gqe_bench.tests.helpers import make_data_info

    db_path = tmp_path / "policies.db3"
    with closing(_GqeConnection(db_path, "test")) as edb:
        edb.executescript(schema._load_base_schema())
        edb.executescript(schema._load_gqe_extension())
        default = insert_data_dimensions(edb, make_data_info(storage_device_kind=storage_kind))
        other = insert_data_dimensions(
            edb,
            make_data_info(storage_device_kind=storage_kind, shmem_policy="future_engine_policy"),
        )
        assert default.data_info_id == other.data_info_id
        assert default.data_info_ext_id != other.data_info_ext_id
        assert (
            insert_data_dimensions(
                edb,
                make_data_info(
                    storage_device_kind=storage_kind, shmem_policy="future_engine_policy"
                ),
            )
            == other
        )
        edb._cursor.execute("INSERT INTO query_info(q_id, q_name) VALUES (1, 'Q1')")
        edb._cursor.execute(
            "INSERT INTO gqe_parameters(p_id, p_sut_info_id) VALUES (1, ?)", (_GQE_SUT_ID,)
        )
        for exp_id, dims, duration in [(1, default, 2.0), (2, other, 1.0)]:
            edb._cursor.execute(
                "INSERT INTO experiment(e_id, e_sut_info_id, e_parameters_id, "
                "e_data_info_id, e_query_info_id, e_sample_size, e_data_info_ext_id) "
                "VALUES (?, ?, 1, ?, 1, 2, ?)",
                (exp_id, _GQE_SUT_ID, dims.data_info_id, dims.data_info_ext_id),
            )
            edb._cursor.executemany(
                "INSERT INTO run(r_experiment_id, r_number, r_duration_s) VALUES (?, ?, ?)",
                [(exp_id, 0, duration), (exp_id, 1, duration)],
            )
        edb._cursor.execute("SELECT * FROM gqe_best_parameters_validated")
        rows = edb._cursor.fetchall()
        assert len(rows) == 1
        row = dict(zip((col[0] for col in edb._cursor.description), rows[0], strict=True))
        assert schema.DataInfo(**schema.DATA_INFO_MAPPING.extract(row)).shmem_policy == (
            "future_engine_policy"
        )
