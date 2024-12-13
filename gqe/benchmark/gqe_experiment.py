# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import asdict, dataclass
from database_benchmarking_tools.experiment import (
    ExperimentConnection,
    ParametersId,
    SutInfoId,
)


@dataclass
class GqeParameters:
    sut_info_id: SutInfoId | None = None
    num_workers: int | None = None
    num_partitions: int | None = None
    join_use_hash_map_cache: bool | None = None
    read_use_zero_copy: bool | None = None


class GqeExperimentConnection(ExperimentConnection):
    def __init__(self, db_path, hostname):
        super().__init__(db_path, hostname)

    def insert_gqe_parameters(self, entry: GqeParameters) -> ParametersId:
        self._cursor.execute(
            " \
            INSERT OR IGNORE INTO gqe_parameters( \
            p_sut_info_id, \
            p_num_workers, \
            p_num_partitions, \
            p_join_use_hash_map_cache,  \
            p_read_use_zero_copy \
            ) \
            VALUES (:sut_info_id, :num_workers, :num_partitions, :join_use_hash_map_cache, :read_use_zero_copy) \
            ",
            asdict(entry),
        )

        self._cursor.execute(
            " \
            SELECT p_id \
            FROM gqe_parameters \
            WHERE p_num_workers = :num_workers \
            AND p_num_partitions = :num_partitions \
            AND p_join_use_hash_map_cache = :join_use_hash_map_cache \
            AND p_read_use_zero_copy = :read_use_zero_copy \
            ",
            asdict(entry),
        )

        return self._cursor.fetchone()[0]
