# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from database_benchmarking_tools.experiment import (
    DataInfoId,
    ExperimentConnection,
    ParametersId,
    SutInfoId,
)
from database_benchmarking_tools import sql_generator


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
    use_partition_pruning: bool

    sut_info_id: SutInfoId | None = None  # Don't set manually


@dataclass
class GqeDataInfoExt:
    _table_name = "gqe_data_info_ext"
    _table_prefix = "de_"
    num_row_groups: int
    compression_format: str
    compression_data_type: str
    compression_chunk_size: int
    zone_map_partition_size: int

    data_info_id: DataInfoId | None = None  # Don't set manually


class GqeExperimentConnection(ExperimentConnection):
    def __init__(self, db_path, hostname):
        super().__init__(db_path, hostname)

    def insert_gqe_parameters(self, entry: GqeParameters) -> ParametersId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)

    def insert_gqe_data_info_ext(self, entry: GqeDataInfoExt) -> DataInfoId:
        sql_generator.insert_or_ignore(self._cursor, entry)
        return sql_generator.select_id(self._cursor, entry)
