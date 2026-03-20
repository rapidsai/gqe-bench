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

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import pandas as pd

import gqe_bench.lib
from database_benchmarking_tools import experiment as exp
from gqe_bench.benchmark.gqe_experiment import (
    GqeDataInfoExt,
    GqeParameters,
)
from gqe_bench.relation import (
    Relation,
)


# Extended Experiment
#
# GQE Bench alters the Experiment table to add a foreign key for the
# GqeDataInfoExt table. This dataclass adds that field in Python.
#
# Note: Consider upstreaming the extension field to
# database-benchmarking-tools, as altering the table implicitly relies on
# `dataclass.asdict` returning all fields of the object.
@dataclass(kw_only=True)
class Experiment(exp.Experiment):
    data_info_ext_id: int


# A unified DataInfo.
@dataclass
class DataInfo(exp.DataInfo, GqeDataInfoExt):
    pass


@dataclass
class EdbInfo:
    sut_info_id: int
    hw_info_id: int
    build_info_id: int


@dataclass
class QueryInfo:
    identifier: str
    root_relation: Relation | gqe_bench.lib.Relation
    reference_solution: str
    validator: Callable[[pd.DataFrame, pd.DataFrame, float], None] = (
        # The lambda transforms the positional `atol` parameter into a named parameter.
        lambda query_result, reference, atol: pd.testing.assert_frame_equal(
            query_result, reference, atol=atol
        )
    )


# In the event of subprocess sandboxing, substrait queries need
# to be created using a Catalog, which calls cuInit(). This
# is used for packaging that information so the QueryInfo object
# can be created in the subprocess.
@dataclass
class QueryInfoContext:
    query_idx: int
    query_str: str
    query_source: str
    reference_file: str
    scale_factor: int
    substrait_file: str
    physical_plan_folder: str


@dataclass(kw_only=True)
class QueryExecutionContext(GqeParameters):
    query_info_ctx: QueryInfoContext


@dataclass
class CatalogContext:
    dataset: str
    storage_kind: str
    num_row_groups: int
    load_data_of_query: int
    identifier_type: gqe_bench.lib.TypeId
    use_opt_char_type: bool
    ddl_file_path: str
    zone_map_partition_size: int
    in_memory_table_compression_format: str
    in_memory_table_compression_chunk_size: int
    in_memory_table_compression_ratio_threshold: float
    in_memory_table_secondary_compression_format: str
    in_memory_table_secondary_compression_ratio_threshold: float
    in_memory_table_secondary_compression_multiplier_threshold: float
    in_memory_table_use_cpu_compression: bool
    in_memory_table_compression_level: int


class QueryError(Enum):
    load_data = 1
    context = 2
    execution = 3
    validation = 4
