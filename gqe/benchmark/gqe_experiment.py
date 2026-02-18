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

from dataclasses import dataclass

from database_benchmarking_tools import sql_generator
from database_benchmarking_tools.experiment import (
    DataInfoId,
    ExperimentConnection,
    ParametersId,
    SutInfoId,
)

import gqe.lib


@dataclass
class GqeParameters:
    _table_name = "gqe_parameters"
    _table_prefix = "p_"
    num_workers: int
    num_partitions: int
    use_overlap_mtx: bool
    join_use_hash_map_cache: bool
    read_use_zero_copy: bool
    join_use_unique_keys: bool
    join_use_perfect_hash: bool
    join_use_mark_join: bool
    use_partition_pruning: bool
    filter_use_like_shift_and: bool
    aggregation_use_perfect_hash: bool

    sut_info_id: SutInfoId | None = None  # Don't set manually


@dataclass
class GqeDataInfoExt:
    _table_name = "gqe_data_info_ext"
    _table_prefix = "de_"
    num_row_groups: int
    compression_format: str
    compression_chunk_size: int
    zone_map_partition_size: int
    compression_ratio_threshold: float
    secondary_compression_format: str
    secondary_compression_ratio_threshold: float
    secondary_compression_multiplier_threshold: float
    use_cpu_compression: bool
    compression_level: int
    data_info_id: DataInfoId | None = None  # Don't set manually


type MetricInfoId = int
type GqeRunExtId = int
type GqeTableStatsId = int
type GqeColumnStatsId = int
type ExperimentId = int
type QueryInfoId = int


@dataclass
class GqeMetricInfo:
    _table_name = "gqe_metric_info"
    _table_prefix = "m_"
    name: str


@dataclass
class GqeRunExt:
    _table_name = "gqe_run_ext"
    _table_prefix = "re_"
    metric_value: float

    experiment_id: ExperimentId | None = None  # Don't set manually
    run_number: int | None = None  # Don't set manually
    metric_info_id: MetricInfoId | None = None  # Don't set manually


@dataclass
class GqeTableStats:
    _table_name = "gqe_table_stats"
    _table_prefix = "ts_"
    table_name: str
    columns: int
    rows: int
    row_groups: int

    data_info_ext_id: DataInfoId
    query_info_id: QueryInfoId

    def __init__(
        self,
        data_info_ext_id: DataInfoId,
        query_info_id: QueryInfoId,
        table_name: str,
        stats: gqe.lib.TableStatistics,
    ):
        self.table_name = table_name
        self.columns = stats.num_columns
        self.rows = stats.num_rows
        self.row_groups = stats.num_row_groups

        self.data_info_ext_id = data_info_ext_id
        self.query_info_id = query_info_id


@dataclass
class GqeColumnStats:
    _table_name = "gqe_column_stats"
    _table_prefix = "cs_"
    gqe_table_stats_id: GqeTableStatsId
    column_name: str
    column_part: str  # "offset" or "char" for string columns, "value" for fixed-width columns.

    compressed_bytes: int
    uncompressed_bytes: int
    slices: int
    compressed_slices: int
    primary_compressed_bytes: int
    secondary_compressed_bytes: int
    num_primary_compressed_slices: int
    num_secondary_compressed_slices: int

    def __init__(
        self,
        gqe_table_stats_id: GqeTableStatsId,
        column_name: str,
        col_idx: int,
        stats: gqe.lib.TableStatistics,
        column_part: str,
    ):
        self.gqe_table_stats_id = gqe_table_stats_id
        self.column_name = column_name
        self.column_part = column_part

        if column_part == "char":
            comp_stats = stats.column_stats[col_idx].get_string_stats().chars_stats
        elif column_part == "offset":
            comp_stats = stats.column_stats[col_idx].get_string_stats().offsets_stats
        else:  # column_part == "value" (fixed-width columns)
            comp_stats = stats.column_stats[col_idx].get_fixed_width_stats()

        self.compressed_bytes = comp_stats.compressed_size
        self.uncompressed_bytes = comp_stats.uncompressed_size
        self.slices = stats.num_row_groups
        self.compressed_slices = comp_stats.num_compressed_row_groups
        self.primary_compressed_bytes = comp_stats.primary_compressed_size
        self.secondary_compressed_bytes = comp_stats.secondary_compressed_size
        self.num_primary_compressed_slices = comp_stats.num_primary_compressed_row_groups
        self.num_secondary_compressed_slices = comp_stats.num_secondary_compressed_row_groups


class GqeExperimentConnection(ExperimentConnection):
    def __init__(self, db_path, hostname):
        super().__init__(db_path, hostname)

    def insert_gqe_parameters(self, entry: GqeParameters) -> ParametersId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def insert_gqe_data_info_ext(self, entry: GqeDataInfoExt) -> DataInfoId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def insert_metric_info(self, entry: GqeMetricInfo) -> MetricInfoId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def insert_gqe_run_ext(self, entry: GqeRunExt) -> GqeRunExtId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def insert_gqe_table_stats(self, entry: GqeTableStats) -> GqeTableStatsId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def insert_gqe_column_stats(self, entry: GqeColumnStats) -> GqeColumnStatsId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)
