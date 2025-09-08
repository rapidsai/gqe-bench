# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import Context
import gqe.lib
from gqe.benchmark.verify import verify_parquet
from gqe.benchmark.gqe_experiment import GqeParameters, GqeDataInfoExt
from gqe.relation import (
    Relation,
    ReadRelation,
    FilterRelation,
    AggregateRelation,
    BroadcastJoinRelation,
)
from gqe.expression import (
    Expression,
    ColumnReference,
    BinaryOpExpression,
    LikeExpr,
    IfThenElseExpr,
    DatePartExpr,
    Cast,
)
from gqe.table_definition import TPCHTableDefinitions

from database_benchmarking_tools import experiment as exp

import importlib.resources
import os
import nvtx
import pandas as pd
import re
from dataclasses import dataclass, asdict, fields
from typing import Optional
from collections.abc import Callable
import copy


# Extended Experiment
#
# gqe-python alters the Experiment table to add a foreign key for the
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
    query_source: str


@dataclass
class QueryInfo:
    identifier: str
    root_relation: Relation | gqe.lib.Relation
    reference_solution: str
    validator: Callable[[pd.DataFrame, pd.DataFrame, float], None] = (
        # The lambda transforms the positional `atol` parameter into a named parameter.
        lambda query_result, reference, atol: pd.testing.assert_frame_equal(
            query_result, reference, atol=atol
        )
    )


@dataclass
class Parameter(GqeParameters):
    pass


# Extract only the fields that belong to the superclass
#
# `asdict(entry)` return all the fields in the object. However, when using
# `exp.sql_generator(entry)` we should only generate the fields that exist in the
# table. This is solved by filtering the fields during upcasting.
def upcast_to_super(obj, super_class):
    field_names = {f.name for f in fields(super_class)}
    super_kwargs = {k: getattr(obj, k) for k in field_names}
    return super_class(**super_kwargs)


def setup_db(edb: exp.ExperimentDB, query_source: str) -> EdbInfo:
    sut_creation_path = importlib.resources.files("gqe.benchmark").joinpath(
        "system_under_test.sql"
    )
    with importlib.resources.as_file(sut_creation_path) as script:
        edb.execute_script(script)

    sut_info_id = edb.insert_sut_info(exp.SutInfo(name="gqe"))
    hw_info_id = edb.insert_hw_info()

    return EdbInfo(sut_info_id, hw_info_id, query_source)


def parse_bool(value: str) -> bool:
    """Parse a string into a Boolean value."""
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('Expected a Boolean value ("true" or "false")')


def parse_scale_factor(path: str) -> int:
    predicate = re.compile(".*(?:sf|SF)([0-9]+)([kK]?).*")
    matches = predicate.match(path)

    if matches is None:
        return None

    scale_factor = int(matches.group(1))
    if matches.group(2):
        scale_factor = scale_factor * 1000

    return scale_factor


def parse_identifier_type(path: str) -> gqe.lib.TypeId:
    """Finds the identifier type in a path to database files."""
    location_parts = re.split(r"_|/", path)
    has_id32 = "id32" in location_parts
    has_id64 = "id64" in location_parts
    if has_id32 and not has_id64:
        return gqe.lib.TypeId.int32
    elif has_id64 and not has_id32:
        return gqe.lib.TypeId.int64
    else:
        raise RuntimeError(f"Can't determine the identifier type of {path}")


def is_valid_identifier_type(
    identifier_type: gqe.lib.TypeId, experiment_suite: str, scale_factor: int
) -> Optional[bool]:
    match experiment_suite:
        case "tpch":
            # int64 is required for scale factors larger than SF 357.
            #
            # Orders has the most primary keys. The number of primary keys is specified as:
            #
            # > O_ORDERKEY unique within [SF * 1,500,000 * 4].
            #
            # Thus, the calculation for the boundary when int32 overflows is:
            #
            # (2^31 - 1) / (4 * 1.5 * 10^6) = 357.91
            return scale_factor < 357 or identifier_type == gqe.lib.TypeId.int64
        case _:
            return None


# Note: Presumably needs to be set before CUDA initialization. CUDA docs don't
# mention when the variable needs to be set. Typically, users will set it
# before launching the program. In our case that's not possible, because the
# program is already running.
def set_eager_module_loading():
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"


def fix_partial_filter_column_references(relation: Relation, query: int):
    """Fix the column references in partial filters.

    Ensures that column references in partial filters of read relations and
    conditions of filter relations refer to the same column X.

    - The column index used in partial filters of read relations refers to the
      position of column X in the base table schema.

    - The column index used in condtions of filter relations refers to the
      position of column X in the list of columns projected by the read
      relation.

    Note: Aggregation relation can also have conditions and are treated the same
    as filter relations.

    """

    # Helper to fix the column references in an expression
    def fix_column_references(expression: Expression, relation: Relation):

        # Leaf condition: Update the column index to refer to the column in the
        # base table schema.
        if isinstance(expression, ColumnReference):
            table = relation.table
            projected_columns = relation.columns
            schema_columns = TPCHTableDefinitions().get_schema(query)[table]
            expression.idx = schema_columns.index(projected_columns[expression.idx])

        # Recursively descent to child expressions
        elif isinstance(expression, BinaryOpExpression):
            fix_column_references(expression.lhs, relation)
            fix_column_references(expression.rhs, relation)
        elif isinstance(expression, (LikeExpr, DatePartExpr, Cast)):
            fix_column_references(expression.input, relation)
        elif isinstance(expression, IfThenElseExpr):
            fix_column_references(expression.if_expr, relation)
            fix_column_references(expression.then_expr, relation)
            fix_column_references(expression.else_expr, relation)

    # Stop recursion when the ReadRelation is reached
    if isinstance(relation, ReadRelation):
        return

    # Fix partial filter of ReadRelation, if it is the direct child of a
    # FilterRelation or AggregateRelation. Also stops recursion.
    elif (
        isinstance(relation, (AggregateRelation, FilterRelation))
        and isinstance(relation.input, ReadRelation)
        and relation.input.partial_filter
    ):
        relation.input.partial_filter = copy.deepcopy(relation.condition)
        fix_column_references(relation.input.partial_filter, relation.input)

    # Recursively descent to child relations
    elif isinstance(relation, BroadcastJoinRelation):
        fix_partial_filter_column_references(relation.left_table, query)
        fix_partial_filter_column_references(relation.right_table, query)
    # Only check instance is not enough as import_module would create different class objects.
    elif relation.__class__.__name__ == 'Q22MarkJoinRelation':
        fix_partial_filter_column_references(relation.customer_table, query)
        fix_partial_filter_column_references(relation.orders_table, query)
    else:
        fix_partial_filter_column_references(relation.input, query)


def run_tpc(
    catalog,
    data: DataInfo,
    query: QueryInfo,
    scale_factor: int,
    parameters: list[Parameter],
    edb: exp.ExperimentDB,
    edb_info: EdbInfo,
    errors: list,
):
    repeat = 6

    data_info_id = edb.insert_data_info(upcast_to_super(data, exp.DataInfo))

    data.data_info_id = data_info_id
    data_info_ext_id = edb.insert_gqe_data_info_ext(
        upcast_to_super(data, GqeDataInfoExt)
    )

    for parameter in parameters:
        debug_mem_usage = bool(os.getenv("GQE_PYTHON_DEBUG_MEM_USAGE", False))

        print(
            f"Running with parameters "
            f"debug_mem_usage={debug_mem_usage}, "
            f"{ parameter }, {data}"
        )

        parameter.sut_info_id = edb_info.sut_info_id

        parameters_id = edb.insert_gqe_parameters(parameter)

        # TODO: use with statement instead?
        context = Context(
            parameter.num_workers,
            parameter.num_partitions,
            data.char_type == "char",
            parameter.use_overlap_mtx,
            parameter.join_use_hash_map_cache,
            parameter.read_use_zero_copy,
            parameter.join_use_unique_keys,
            parameter.join_use_perfect_hash,
            data.compression_format,
            data.compression_data_type,
            data.compression_chunk_size,
            parameter.use_partition_pruning,
            data.zone_map_partition_size,
            debug_mem_usage,
        )

        print(f"Running {query.identifier}...")

        experiment_id = edb.insert_experiment(
            Experiment(
                sut_info_id=edb_info.sut_info_id,
                parameters_id=parameters_id,
                hw_info_id=edb_info.hw_info_id,
                build_info_id=None,
                data_info_id=data_info_id,
                data_info_ext_id=data_info_ext_id,
                name=query.identifier,
                suite="TPC-H",
                scale_factor=scale_factor,
                query_source=edb_info.query_source,
            )
        )

        for count in range(repeat):
            out_file = f"{query.identifier}_out.parquet"

            with nvtx.annotate(f"Run {query.identifier}"):
                try:
                    elapsed_time = context.execute(
                        catalog, query.root_relation, out_file
                    )
                except Exception as error:
                    print(error)
                    break

            try:
                verify_parquet(out_file, query.reference_solution, query.validator)
            except Exception as error:
                print(error)
                errors.append((query.identifier, parameter))
                break

            edb.insert_run(
                exp.Run(
                    experiment_id=experiment_id,
                    number=count,
                    nvtx_marker=None,
                    duration_s=elapsed_time / 1000,
                )
            )

        del context
