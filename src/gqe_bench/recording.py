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

"""Experiment database setup and recording.

Creates the SQLite experiment database, inserts system/data/query
dimension rows, and records experiment + failed-run entries. Schema
metadata (table definitions, column mappings, type classes) comes
from schema.py -- this module only handles writes.

Plugin protocol:
    Env-var name constants (``GQE_BENCH_DB_ENV``,
    ``GQE_BENCH_TIME_BREAKDOWN_ENV``, ``GQE_BENCH_CUPTI_METRICS_ENV``)
    and the plugin ``.so`` location (``PLUGIN``) come from
    ``gqe_bench._artifacts``, which CMake generates from the same variables
    that feed the C++ ``env.hpp`` via ``configure_file``. Drift between C++
    and Python is structurally impossible.

    The plugin reads env vars exactly once at library-constructor time,
    under LD_PRELOAD into ``gqe_node_manager`` / ``gqe_task_manager``.
    Setting a variable after the subprocess has started has no effect.
    ``SweepContext.env`` is authoritative: plugin env vars are NOT
    inherited from the parent shell. Empty string means "off" for the
    boolean / list toggles. ``LD_PRELOAD`` is populated by
    ``_build_plugin_env`` from the plugin resource location.
"""

import dataclasses
import importlib.resources
import os
import socket
from argparse import Namespace
from pathlib import Path
from typing import Any, NamedTuple

import database_benchmarking_tools.experiment as exp
from database_benchmarking_tools import sql_generator
from database_benchmarking_tools.experiment import ExperimentConnection, ExperimentDB
from database_benchmarking_tools.utility import generate_db_path
from gqe_bench._agg_protocol_sizes import K_HEADER_BYTES, K_SLOT_BYTES
from gqe_bench._artifacts import (
    GQE_BENCH_CUPTI_METRICS_ENV,
    GQE_BENCH_DB_ENV,
    GQE_BENCH_LOG_LEVEL_ENV,
    GQE_BENCH_NUM_RANKS_ENV,
    GQE_BENCH_SHM_NAME_ENV,
    GQE_BENCH_SHM_SIZE_ENV,
    GQE_BENCH_TIME_BREAKDOWN_ENV,
    PLUGIN,
)
from gqe_bench._build_info import GIT_BRANCH, GIT_IS_DIRTY, GIT_REVISION
from gqe_bench.schema import (
    DATA_INFO_MAPPING,
    QUERY_NAME_PREFIX,
    QUERY_PARAMS_MAPPING,
    DataInfo,
    Query,
    QueryParams,
    get_tables,
)

_SUT_NAME = "gqe"
# --log-level values translated to spdlog level names for the plugin. Total over
# the levels arguments.py accepts: spdlog's from_str() maps anything it does not
# recognise to "off", so a missing key would silence the plugin rather than fail.
_SPDLOG_LEVELS: dict[str, str] = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warning",
    "ERROR": "error",
    "QUIET": "off",
}
_GQE_EXTENSION_DDL = (
    importlib.resources.files("gqe_bench.sql")
    .joinpath("system_under_test.sql")
    .read_text(encoding="utf-8")
)


def _make_insert_class(table_name: str) -> type:
    """Generate a dataclass for sql_generator from PRAGMA columns."""
    t = get_tables()[table_name]
    fields = [(col[len(t.prefix) :], Any) for col in t.columns if col != f"{t.prefix}id"]
    return dataclasses.make_dataclass(
        f"_{table_name}",
        fields,
        namespace={"_table_name": table_name, "_table_prefix": t.prefix},
    )


_GqeParameters = _make_insert_class("gqe_parameters")
_GqeDataInfoExt = _make_insert_class("gqe_data_info_ext")


@dataclasses.dataclass(frozen=True, kw_only=True)
class _GqeExperiment(exp.Experiment):
    """Experiment with GQE extension column (added by ALTER TABLE in the DDL)."""

    data_info_ext_id: int


class _GqeConnection(ExperimentConnection):
    """ExperimentConnection extended with GQE-specific insert methods."""

    def executescript(self, sql: str) -> None:
        """Execute a multi-statement SQL string."""
        self._cursor.executescript(sql)

    def insert_gqe_parameters(self, entry: _GqeParameters) -> int:
        """Insert or ignore a gqe_parameters row; return its id."""
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def insert_gqe_data_info_ext(self, entry: _GqeDataInfoExt) -> int:
        """Insert or ignore a gqe_data_info_ext row; return its id."""
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)


class SystemIds(NamedTuple):
    """System-level dimension-row IDs (system-under-test, hardware, build)."""

    sut_info_id: int
    hw_info_id: int
    build_info_id: int


@dataclasses.dataclass(frozen=True)
class DataDimensions:
    """IDs produced by ``insert_data_dimensions`` for one DataInfo.

    Passed opaquely from ``_run_group`` into ``record_experiment``; callers
    outside ``recording.py`` have no reason to inspect the fields.
    """

    data_info_id: int
    data_info_ext_id: int


@dataclasses.dataclass(frozen=True)
class SweepContext:
    """Sweep-scoped execution context — invariant across every experiment
    and every run in the sweep.

    Produced once by ``setup_db(args)`` from a mix of setup-time side effects
    (DB creation, system dimension inserts) and parameters read off ``args``.
    Threaded through ``runner.main()`` into ``_run_group`` and ``record_experiment``.

    Attributes:
        db_mgr:        Connection manager for the experiment database.
        system_ids:    Dimension-row IDs assigned during setup (sut, hw, build).
        gpu_info_ids:  ``g_id`` of every GPU this sweep will use, ordered by
                       CUDA device index. List index ≡ rank ≡ CUDA index.
                       Used by ``record_experiment`` to insert one
                       ``experiment_gpu`` row per (experiment, rank). The
                       plugin does not consume this — rank 0 resolves its
                       own ``gpu_info_id`` mapping in C++ via
                       ``cuDeviceGetUuid`` per cuda index and a SELECT
                       on ``gpu_info`` keyed by ``g_gpu_uuid``.
        env:           Authoritative subprocess environment for processes that
                       load libgqe_bench.so. Callers MUST pass this to
                       ``GqeServer(env=...)`` (merged with ``os.environ`` if
                       general inheritance is desired). Plugin-specific env vars
                       are NOT inherited from the parent shell — the plugin only
                       honors the keys this dict sets; empty string means off.
        suite_name:    Benchmark suite identifier (e.g. "TPC-H"). Written into
                       ``query_info.suite`` for every experiment.
        repeat:        Per-experiment run count. Controls ``range(repeat)`` in
                       the runner loop and ``experiment.sample_size`` in the DB.
    """

    db_mgr: ExperimentDB
    system_ids: SystemIds
    gpu_info_ids: list[int]
    env: dict[str, str]
    suite_name: str
    repeat: int

    def __enter__(self) -> "SweepContext":
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback) -> bool:
        """Close the database after every writer is gone.

        Nested outside the GqeSession, so the task managers whose plugin holds
        its own handle on the same file have exited by the time this runs. That
        makes this the last connection, which is what checkpoints the WAL and
        removes the -wal and -shm sidecars.
        """
        with self.db_mgr:
            pass
        return False


def _build_plugin_env(db_path: str, args: Namespace) -> dict[str, str]:
    """Build the authoritative env dict for libgqe_bench.so subprocesses.

    Sets ``LD_PRELOAD`` to the package-resolved plugin path and every
    plugin-protocol env var explicitly (empty string when off). Raises
    ``ArtifactMissing`` via ``PLUGIN.require()`` if the plugin isn't present —
    the runner path cannot proceed without it, so the error surfaces here
    with actionable context (build flag name) rather than as a silent
    no-op subprocess downstream.

    Multi-GPU env vars are populated unconditionally; the plugin's
    ``execute_plan_observer::try_attach`` collapses to a single-rank
    fast path when ``num_gpus == 1`` and the shm segment carries a
    single slot. The actual segment is created/unlinked by
    ``GqeSession.__enter__`` / ``__exit__`` around the server lifetime.
    """
    num_ranks = max(int(args.num_gpus), 1)
    shm_name = f"/gqe_bench_agg_{os.getpid()}"
    shm_size = K_HEADER_BYTES + num_ranks * K_SLOT_BYTES
    return {
        "LD_PRELOAD": str(PLUGIN.require()),
        GQE_BENCH_DB_ENV: db_path,
        GQE_BENCH_TIME_BREAKDOWN_ENV: "1" if args.time_breakdown else "",
        GQE_BENCH_CUPTI_METRICS_ENV: ",".join(args.cupti_metrics) if args.cupti_metrics else "",
        GQE_BENCH_SHM_NAME_ENV: shm_name,
        GQE_BENCH_NUM_RANKS_ENV: str(num_ranks),
        GQE_BENCH_SHM_SIZE_ENV: str(shm_size),
        GQE_BENCH_LOG_LEVEL_ENV: _SPDLOG_LEVELS[args.log_level],
    }


def setup_db(args: Namespace) -> SweepContext:
    """Create the experiment DB, insert system-level dimension rows, build
    the plugin-protocol env dict, and capture the sweep-scoped parameters.

    Returns a ``SweepContext`` bundling everything invariant across the
    sweep: DB manager, system-level dimension IDs, the plugin subprocess
    environment, the suite name, and the per-experiment repeat count. The
    caller threads ``sweep.env`` into ``GqeServer(env=...)`` (typically
    merged with ``os.environ``).
    """
    hostname = socket.gethostname()
    db_path = (
        str(args.output) if args.output else generate_db_path(_SUT_NAME, args.suite_name, hostname)
    )

    db_file = Path(db_path)
    if db_file.exists():
        db_file.unlink()

    db_mgr = ExperimentDB(db_path, hostname).set_connection_type(_GqeConnection)
    db_mgr.create_experiment_db()

    with db_mgr as edb:
        edb.executescript(_GQE_EXTENSION_DDL)
        edb.commit()

        edb._cursor.execute("PRAGMA journal_mode=WAL")
        edb.commit()

        sut_id = edb.get_sut_info_id(exp.SutInfo(name=_SUT_NAME))
        hw_id = edb.insert_hw_info()
        build_id = edb.insert_build_info(
            exp.BuildInfo(
                revision=GIT_REVISION,
                branch=GIT_BRANCH,
                is_dirty=GIT_IS_DIRTY,
            )
        )

        # One ``gpu_info`` row per CUDA device index this sweep will use.
        # The list index is the rank; the UUID stored on each row comes
        # from ``cuDeviceGetUuid(cuda_index)``, which honors
        # ``CUDA_VISIBLE_DEVICES`` filtering / MIG / reordering — unlike
        # NVML's index, which does not.
        gpu_info_ids = [
            edb.insert_gpu_info(hw_id, cuda_index) for cuda_index in range(args.num_gpus)
        ]
        edb.commit()

    return SweepContext(
        db_mgr=db_mgr,
        system_ids=SystemIds(sut_id, hw_id, build_id),
        gpu_info_ids=gpu_info_ids,
        env=_build_plugin_env(db_path, args),
        suite_name=args.suite_name,
        repeat=args.repeat,
    )


def _upcast_to_super(obj: object, target_class: type) -> object:
    """Build a target_class instance from matching field names on obj."""
    kwargs = {f.name: getattr(obj, f.name) for f in dataclasses.fields(target_class)}
    return target_class(**kwargs)


def insert_data_dimensions(
    edb: _GqeConnection,
    data_info: DataInfo,
) -> DataDimensions:
    """Insert base data_info and gqe_data_info_ext rows. Returns both IDs bundled."""
    di_id = edb.insert_data_info(_upcast_to_super(data_info, exp.DataInfo))

    ext_kwargs = DATA_INFO_MAPPING.to_insert_kwargs(data_info, data_info_id=di_id)
    di_ext_id = edb.insert_gqe_data_info_ext(_GqeDataInfoExt(**ext_kwargs))
    return DataDimensions(data_info_id=di_id, data_info_ext_id=di_ext_id)


def record_experiment(
    edb: _GqeConnection,
    dims: DataDimensions,
    query: Query,
    qp: QueryParams,
    sweep: SweepContext,
) -> int:
    """Insert query dimensions and experiment record. Returns experiment_id.

    ``query_info.q_name`` is recorded with ``QUERY_NAME_PREFIX`` (``Q13_fused``)
    while ``Query.name`` stays bare (``13_fused``); ``_query_validated_params``
    removes the prefix when reading rows back out.
    """
    params_kwargs = QUERY_PARAMS_MAPPING.to_insert_kwargs(
        qp,
        sut_info_id=sweep.system_ids.sut_info_id,
    )
    params_id = edb.insert_gqe_parameters(_GqeParameters(**params_kwargs))

    qi_id = edb.insert_query_info(
        exp.QueryInfo(
            name=f"{QUERY_NAME_PREFIX}{query.name}",
            suite=sweep.suite_name,
            source=str(query.source),
        )
    )

    experiment_id = edb.insert_experiment(
        _GqeExperiment(
            sut_info_id=sweep.system_ids.sut_info_id,
            parameters_id=params_id,
            build_info_id=sweep.system_ids.build_info_id,
            data_info_id=dims.data_info_id,
            query_info_id=qi_id,
            sample_size=sweep.repeat,
            data_info_ext_id=dims.data_info_ext_id,
        )
    )

    # Tag the experiment with every GPU it will run on. List index ≡ rank ≡
    # CUDA device index, by construction in ``setup_db``.
    for cuda_index, gpu_info_id in enumerate(sweep.gpu_info_ids):
        edb.insert_experiment_gpu(
            exp.ExperimentGpu(
                experiment_id=experiment_id,
                gpu_info_id=gpu_info_id,
                cuda_index=cuda_index,
            )
        )

    return experiment_id


def delete_run_data(
    edb: ExperimentConnection,
    experiment_id: int,
    run_number: int,
) -> None:
    """Remove run + associated profiling rows for the given (experiment_id, run_number).

    Used on query failure to clean up rows the plugin committed before
    Python observed the validation/execution failure. DELETEs against all
    plugin-written tables are harmless no-ops if the plugin didn't write for
    this run. Activity tables FK to ``run``, so they must be deleted before
    ``run``.
    """
    edb._cursor.execute(
        "DELETE FROM gqe_run_ext WHERE re_experiment_id = ? AND re_run_number = ?",
        (experiment_id, run_number),
    )
    edb._cursor.execute(
        "DELETE FROM gqe_run_time_breakdown WHERE tb_experiment_id = ? AND tb_run_number = ?",
        (experiment_id, run_number),
    )
    edb._cursor.execute(
        "DELETE FROM gqe_run_cupti_kernel_activity "
        "WHERE ka_experiment_id = ? AND ka_run_number = ?",
        (experiment_id, run_number),
    )
    edb._cursor.execute(
        "DELETE FROM gqe_run_cupti_memcpy_activity "
        "WHERE mca_experiment_id = ? AND mca_run_number = ?",
        (experiment_id, run_number),
    )
    edb._cursor.execute(
        "DELETE FROM gqe_run_cupti_marker_activity "
        "WHERE mra_experiment_id = ? AND mra_run_number = ?",
        (experiment_id, run_number),
    )
    edb._cursor.execute(
        "DELETE FROM gqe_run_cupti_mem_decompress_activity "
        "WHERE mda_experiment_id = ? AND mda_run_number = ?",
        (experiment_id, run_number),
    )
    edb._cursor.execute(
        "DELETE FROM run WHERE r_experiment_id = ? AND r_number = ?",
        (experiment_id, run_number),
    )


def record_failed_run(
    edb: ExperimentConnection,
    experiment_id: int,
    run_number: int,
    error_msg: str,
) -> None:
    """Insert a FailedRun record for a query that errored or failed validation."""
    edb.insert_failed_run(
        exp.FailedRun(
            experiment_id=experiment_id,
            number=run_number,
            error_msg=error_msg,
        )
    )


def replace_run_with_failure(
    edb: ExperimentConnection,
    experiment_id: int,
    run_number: int,
    error_msg: str,
) -> None:
    """Replace any plugin-committed ``run`` row for this attempt with a failure record."""
    delete_run_data(edb, experiment_id, run_number)
    record_failed_run(edb, experiment_id, run_number, error_msg)
