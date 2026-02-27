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

import importlib.resources
import platform
import sqlite3
from dataclasses import dataclass

from database_benchmarking_tools import hardware_info, sql_generator, utility

type BuildInfoId = int
type DataInfoId = int
type ExperimentId = int
type HwInfoId = int
type ParametersId = int
type SutInfoId = int
type QueryInfoId = int


@dataclass
class BuildInfo:
    _table_name = "build_info"
    _table_prefix = "b_"
    version: str | None = None
    revision: str | None = None
    branch: int | None = None
    is_dirty: int | None = None
    commit_timestamp: int | None = None
    compiler_flags: str | None = None


@dataclass
class DataInfo:
    _table_name = "data_info"
    _table_prefix = "d_"
    storage_device_kind: str | None = None
    format: str | None = None
    location: str | None = None
    not_null: bool | None = None
    identifier_type: str | None = None
    char_type: str | None = None
    decimal_type: str | None = None
    scale_factor: float | None = None


@dataclass
class HwInfo:
    _table_name = "hw_info"
    _table_prefix = "h_"
    hostname: str | None = None
    cpu_arch: str | None = None
    cpu_model_name: str | None = None
    cpu_clock_mhz: str | None = None
    cpu_physical_cores: str | None = None
    cpu_logical_cores: str | None = None
    gpu_product_name: str | None = None
    nvidia_driver_version: str | None = None
    cuda_version: str | None = None
    gpu_pcie_link_generation: int | None = None
    gpu_cuda_cores: int | None = None
    gpu_max_clock_sm_mhz: int | None = None
    gpu_max_clock_memory_mhz: int | None = None
    gpu_ecc_errors: int | None = None


@dataclass
class Experiment:
    _table_name = "experiment"
    _table_prefix = "e_"
    sut_info_id: SutInfoId | None = None
    parameters_id: ParametersId | None = None
    hw_info_id: HwInfoId | None = None
    build_info_id: BuildInfoId | None = None
    data_info_id: DataInfoId | None = None
    query_info_id: QueryInfoId | None = None
    sample_size: int | None = None


@dataclass
class Run:
    _table_name = "run"
    _table_prefix = "r_"
    experiment_id: ExperimentId | None = None
    number: int | None = None
    nvtx_marker: int | None = None
    duration_s: float | None = None


@dataclass
class FailedRun:
    _table_name = "failed_run"
    _table_prefix = "fr_"
    experiment_id: ExperimentId | None = None
    number: int | None = None
    error_msg: str | None = None


@dataclass
class SutInfo:
    _table_name = "sut_info"
    _table_prefix = "s_"
    name: str | None = None


@dataclass
class QueryInfo:
    _table_name = "query_info"
    _table_prefix = "q_"
    name: str | None = None
    suite: str | None = None
    source: str | None = None


class ExperimentConnection:
    def __init__(self, db_path, hostname):
        self._conn = sqlite3.connect(db_path)
        self._cursor = self._conn.cursor()
        self._db_path = db_path
        self._hostname = hostname

    def __del__(self):
        self._cursor.close()
        self._conn.close()

    def execute_script(self, script_path: str):
        with open(script_path, mode="r") as script_sql:
            self._cursor.executescript(script_sql.read())

    def commit(self):
        self._analyze()
        self._conn.commit()

    def hostname(self) -> str:
        return self._hostname

    def insert_build_info(self, entry: BuildInfo) -> BuildInfoId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def insert_data_info(self, entry: DataInfo) -> DataInfoId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def insert_hw_info(self) -> HwInfoId:
        gpu_id = 0

        cpu_info = hardware_info.CpuInfo()
        gpu_info = hardware_info.GpuInfo()

        entry = HwInfo(
            hostname=utility.get_hostname(self._hostname),
            cpu_arch=platform.machine(),
            cpu_model_name=cpu_info.model_name(),
            cpu_clock_mhz=int(cpu_info.cpu_mhz()),
            cpu_physical_cores=cpu_info.cpu_physical_cores(),
            gpu_product_name=gpu_info.device_product_name(gpu_id),
            nvidia_driver_version=gpu_info.system_driver_version(),
            cuda_version=gpu_info.cuda_driver_version(),
            gpu_pcie_link_generation=gpu_info.pcie_link_generation(gpu_id),
            gpu_cuda_cores=gpu_info.gpu_cores(gpu_id),
            gpu_max_clock_sm_mhz=gpu_info.max_sm_clock(gpu_id),
            gpu_max_clock_memory_mhz=gpu_info.max_memory_clock(gpu_id),
            gpu_ecc_errors=gpu_info.total_ecc_errors(gpu_id),
        )

        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def insert_sut_info(self, entry: SutInfo) -> SutInfoId:
        return sql_generator.select_id(self._cursor, entry)

    def insert_experiment(self, entry: Experiment) -> ExperimentId:
        experiment_id = sql_generator.insert(self._cursor, entry)

        # Infer from experiment without any runs that the first run
        # failed and experiment was skipped.
        self._conn.commit()

        return experiment_id

    def insert_run(self, entry: Run):
        sql_generator.insert_natural_key(self._cursor, entry)

        # Ensure that each successful run is written to DB.
        self._conn.commit()

    def insert_failed_run(self, entry: FailedRun):
        sql_generator.insert_natural_key(self._cursor, entry)
        # Ensure that each successful run is written to DB.
        self._conn.commit()

    def insert_query_info(self, entry: QueryInfo) -> QueryInfoId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def _analyze(self):
        self._cursor.execute("ANALYZE")


class ExperimentDB:
    _db_path = None
    _hostname = None
    _connection_type = ExperimentConnection

    def __init__(self, db_path: str, hostname: str | None = None):
        self._db_path = db_path
        self._hostname = hostname

    def __enter__(self):
        self.__conn = self.connect()

        return self.__conn

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.__conn.commit()
        del self.__conn

        return False

    def create_experiment_db(self):
        conn = sqlite3.connect(self._db_path)
        measurement_schema_path = importlib.resources.files(
            "database_benchmarking_tools.sql"
        ).joinpath("create_experiment_db.sql")

        with importlib.resources.as_file(measurement_schema_path) as measurements_schema:
            with open(measurements_schema, mode="r") as schema_sql:
                conn.executescript(schema_sql.read())
        conn.close()

    # Abstract factor pattern for Connection used by `connect()`
    def set_connection_type(self, connection_type):
        self._connection_type = connection_type
        return self

    def connect(self):
        return self._connection_type(self._db_path, self._hostname)
