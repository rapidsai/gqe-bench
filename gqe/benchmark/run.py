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

import argparse
import importlib.resources

# Alias multiprocessing due to namespace class in GQE
import multiprocessing as subprocessing
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import Optional

import nvtx
import pandas as pd
from database_benchmarking_tools import experiment as exp
from database_benchmarking_tools.experiment import ExperimentDB

import gqe.lib
from gqe import Catalog, Context, MultiProcessContext, optimization_parameters
from gqe.benchmark.gqe_experiment import (
    GqeColumnStats,
    GqeDataInfoExt,
    GqeExperimentConnection,
    GqeMetricInfo,
    GqeParameters,
    GqeRunExt,
    GqeTableStats,
)
from gqe.benchmark.validate import validate_parquet
from gqe.expression import (
    BinaryOpExpression,
    Cast,
    ColumnReference,
    DatePartExpr,
    Expression,
    IfThenElseExpr,
    LikeExpr,
)
from gqe.relation import (
    BroadcastJoinRelation,
    ReadRelation,
    Relation,
    ShuffleJoinRelation,
)
from gqe.table_definition import TPCHTableDefinitions


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
    build_info_id: int


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
    identifier_type: gqe.lib.TypeId
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


# Extract only the fields that belong to the superclass
#
# `asdict(entry)` return all the fields in the object. However, when using
# `exp.sql_generator(entry)` we should only generate the fields that exist in the
# table. This is solved by filtering the fields during upcasting.
def upcast_to_super(obj, super_class):
    field_names = {f.name for f in fields(super_class)}
    super_kwargs = {k: getattr(obj, k) for k in field_names}
    return super_class(**super_kwargs)


def setup_db(edb: exp.ExperimentDB) -> EdbInfo:
    sut_creation_path = importlib.resources.files("gqe.benchmark").joinpath("system_under_test.sql")
    with importlib.resources.as_file(sut_creation_path) as script:
        edb.execute_script(script)

    sut_info_id = edb.insert_sut_info(exp.SutInfo(name="gqe"))
    hw_info_id = edb.insert_hw_info()
    build_info_id = edb.insert_build_info(
        exp.BuildInfo(
            revision=gqe.lib.libgqe_commit,
            is_dirty=gqe.lib.libgqe_is_dirty,
            branch=gqe.lib.libgqe_branch,
        )
    )
    return EdbInfo(sut_info_id, hw_info_id, build_info_id)


def parse_bool(value: str) -> bool:
    """Parse a string into a Boolean value."""
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('Expected a Boolean value ("true" or "false")')


def parse_scale_factor(path: str) -> float:
    predicate = re.compile(r".*(?:sf|SF)[-_]?([0-9.]+)([kK]?).*")
    matches = predicate.match(path)

    if matches is None:
        return None

    scale_factor = float(matches.group(1))
    if matches.group(2):
        scale_factor = scale_factor * 1000

    return scale_factor


def parse_identifier_type(path: str) -> gqe.lib.TypeId:
    """Finds the identifier type in a path to database files."""
    location_parts = re.split(r"_|/", path)
    has_id32 = "id32" in location_parts
    has_id64 = "id64" in location_parts
    if has_id32 and not has_id64:
        identifier_type = gqe.lib.TypeId.int32
    elif has_id64 and not has_id32:
        identifier_type = gqe.lib.TypeId.int64
    else:
        scale_factor = parse_scale_factor(path)
        identifier_type = gqe.lib.TypeId.int32 if scale_factor < 357 else gqe.lib.TypeId.int64
    return identifier_type


def identifier_type_to_sql(identifier_type: gqe.lib.TypeId) -> str:
    """Convert a TypeId to SQL-style type string."""
    type_to_sql = {
        gqe.lib.TypeId.int32: "int",
        gqe.lib.TypeId.int64: "bigint",
    }
    return type_to_sql[identifier_type]


def sql_to_identifier_type(sql_type: str) -> gqe.lib.TypeId:
    """Convert a SQL-style type string to TypeId."""
    sql_to_type = {
        "int": gqe.lib.TypeId.int32,
        "bigint": gqe.lib.TypeId.int64,
    }
    return sql_to_type[sql_type]


# FIXME: handle this for other datasets
def is_valid_identifier_type(
    identifier_type: gqe.lib.TypeId, experiment_suite: str, scale_factor: int
) -> Optional[bool]:
    match experiment_suite:
        case "TPC-H":
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


def get_query_validator(query_object):
    try:
        query_validator = getattr(query_object, "validate_query")
        # Check that the validator is a callable method.
        if not callable(query_validator):
            raise TypeError("Expected the query validator to be a callable method")
        else:
            return query_validator

    except AttributeError:
        # The query doesn't have a validator.
        return None


# Note: Presumably needs to be set before CUDA initialization. CUDA docs don't
# mention when the variable needs to be set. Typically, users will set it
# before launching the program. In our case that's not possible, because the
# program is already running.
def set_eager_module_loading():
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"


def boost_shared_memory_pool_size(
    input_pool_size: int, scale_factor: int, load_all_data: bool
) -> int:
    if input_pool_size:
        return input_pool_size * 1024 * 1024 * 1024
    # Found through experiments 50 GBs fits SF100
    if not load_all_data:
        return int(scale_factor * (1 / 2) * 1024 * 1024 * 1024)
    # Found through experiments 70GBs fits SF100
    if load_all_data:
        return int(scale_factor * (7 / 10) * 1024 * 1024 * 1024)


def fix_partial_filter_column_references(
    relation: Relation, query: int, fixed_read_relations: list[int] | None = None
):
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
            table_columns = TPCHTableDefinitions().get_schema(0)[table]
            loaded_columns = TPCHTableDefinitions().get_schema(query)[table]
            column_name = table_columns[expression.idx]
            expression.idx = loaded_columns.index(column_name)

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

    if fixed_read_relations is None:
        fixed_read_relations = []

    # Fix partial filter of ReadRelation
    if isinstance(relation, ReadRelation):
        # Some query plans reuse operator subtrees in multiple locations, e.g.,
        # Q18_opt. We have to make sure that each ReadRelation of these subtrees
        # is only processed once.
        if relation.partial_filter and id(relation) not in fixed_read_relations:
            fixed_read_relations.append(id(relation))
            fix_column_references(relation.partial_filter, relation)

    # Recursively descent to child relations
    elif isinstance(relation, (BroadcastJoinRelation, ShuffleJoinRelation)):
        fix_partial_filter_column_references(relation.left_table, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.right_table, query, fixed_read_relations)
    # Only check instance is not enough as import_module would create different class objects.
    elif relation.__class__.__name__ == "Q10FusedProbesJoinMapBuildRelation":
        fix_partial_filter_column_references(relation.build_side_table, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q10FusedProbesJoinMultimapBuildRelation":
        fix_partial_filter_column_references(relation.build_side_table, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q10FusedProbesJoinProbeRelation":
        fix_partial_filter_column_references(
            relation.o_custkey_to_row_indices_multimap, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.n_nationkey_to_row_index_map, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.join_orders_lineitem_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(relation.nation_table, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.customer_table, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q10SortLimitRelation":
        fix_partial_filter_column_references(relation.input_table, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q10UniqueKeyInnerJoinBuildRelation":
        fix_partial_filter_column_references(relation.build_side_table, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q10UniqueKeyInnerJoinProbeRelation":
        fix_partial_filter_column_references(relation.build_side_map, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.build_side_table, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.probe_side_table, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q13GroupjoinProbeRelation":
        fix_partial_filter_column_references(relation.groupjoin_build, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.orders, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q13FusedFilterProbeRelation":
        fix_partial_filter_column_references(relation.groupjoin_build, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.orders, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q16FusedFilterJoinRelation":
        fix_partial_filter_column_references(relation.supplier_table, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.part_table, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.partsupp_table, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q21LeftAntiJoinProbeRelation":
        fix_partial_filter_column_references(relation.left_table, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.right_table, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q21LeftSemiJoinProbeRelation":
        fix_partial_filter_column_references(relation.left_table, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.right_table, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q21LeftAntiJoinRetrieveRelation":
        fix_partial_filter_column_references(relation.left_table, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.probe, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q21LeftSemiJoinRetrieveRelation":
        fix_partial_filter_column_references(relation.left_table, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.probe, query, fixed_read_relations)
    elif relation.__class__.__name__ == "Q22MarkJoinRelation":
        fix_partial_filter_column_references(relation.customer_table, query, fixed_read_relations)
        fix_partial_filter_column_references(relation.orders_table, query, fixed_read_relations)
    else:
        fix_partial_filter_column_references(relation.input, query, fixed_read_relations)


def log_physical_plan(query_str: str, relation: gqe.lib.Relation, folder_path: str):
    file_path = os.path.join(folder_path, query_str + "_plan.json")
    gqe.lib.log_physical_plan(relation, file_path)


def _get_query_info(
    query_info_ctx: QueryInfoContext,
    load_all_data: bool,
    storage_kind: str,
    table_definitions: TPCHTableDefinitions,
    catalog: gqe.Catalog,
    multiprocess_runtime_context: gqe.lib.MultiProcessRuntimeContext,
):
    if query_info_ctx.query_source == "handcoded":
        query_identifier = "tpch_q" + query_info_ctx.query_str
        module = importlib.import_module(f"gqe.benchmark.{query_identifier}")
        query_object = getattr(module, query_identifier)(scale_factor=query_info_ctx.scale_factor)
        root_relation = query_object.root_relation(table_definitions)
        query = QueryInfo(
            f"Q{query_info_ctx.query_str}", root_relation, query_info_ctx.reference_file
        )
        # Fix partial filter column references for handcoded queries
        if not load_all_data and (storage_kind != "parquet_file"):
            fix_partial_filter_column_references(root_relation, query_info_ctx.query_idx)
        if validator := get_query_validator(query_object):
            query.validator = validator

        if query_info_ctx.physical_plan_folder:
            root_relation.log_physical_plan(
                f"Q{query_info_ctx.query_str}", query_info_ctx.physical_plan_folder
            )
    elif (
        query_info_ctx.query_source == "substrait"
        or query_info_ctx.query_source == "custom_substrait"
    ):
        root_relation = catalog.load_substrait(
            query_info_ctx.substrait_file, True, multiprocess_runtime_context
        )
        query = QueryInfo(
            f"Q{query_info_ctx.query_str}", root_relation, query_info_ctx.reference_file
        )

        if query_info_ctx.physical_plan_folder:
            log_physical_plan(
                f"Q{query_info_ctx.query_str}",
                root_relation,
                query_info_ctx.physical_plan_folder,
            )

    return query


def print_mp(message, verbose):
    if verbose:
        print(message)


def run_suite(
    cat_ctx: CatalogContext,
    data: DataInfo,
    scale_factor: int,
    parameters: list[QueryExecutionContext],
    edb_file: str,
    edb_info: EdbInfo,
    cupti_metrics: list[str] | None,
    errors: list,
    invalid_results: list,
    repeat: int,
    is_root_rank: bool,
    is_mp: bool,
    multiprocess_runtime_context: gqe.lib.MultiProcessRuntimeContext,
    validate_results: bool,
    validate_dir: str,
    suite_name: str,
    quiet: bool = False,
    pipe: subprocessing.Pipe = None,
):
    # only send DB to root rank so we will get error if there is a logic mistake
    if is_root_rank:
        gqe_host = "localhost"
        edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(GqeExperimentConnection)
        with edb_config as edb:
            _run_suite(
                cat_ctx,
                data,
                scale_factor,
                parameters,
                edb,
                edb_info,
                cupti_metrics,
                errors,
                invalid_results,
                repeat,
                is_root_rank,
                is_mp,
                multiprocess_runtime_context,
                validate_results,
                validate_dir,
                suite_name,
                quiet,
                pipe,
            )
    else:
        _run_suite(
            cat_ctx,
            data,
            scale_factor,
            parameters,
            None,
            edb_info,
            None,
            errors,
            invalid_results,
            repeat,
            is_root_rank,
            is_mp,
            multiprocess_runtime_context,
            validate_results,
            validate_dir,
            suite_name,
            quiet,
            pipe,
        )


def pipe_send(pipe: subprocessing.Pipe, status: bool):
    if pipe is not None:
        pipe.send(status)


# if cuda error matches any of these regex, it's unrecoverable and we need a new process
def is_unrecoverable_error(e):
    error = f"{e}"
    # This should catch all "cudaError" types. Leaving structure in case we have non-cuda errors of this class.
    # https://docs.nvidia.com/cuda//cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038
    messages = [
        "cudaError",
    ]
    for message in messages:
        if message in error:
            return True
    return False


def _run_suite(
    cat_ctx: CatalogContext,
    data: DataInfo,
    scale_factor: int,
    parameters: list[QueryExecutionContext],
    edb: exp.ExperimentDB,
    edb_info: EdbInfo,
    cupti_metrics: list[str] | None,
    errors: list,
    invalid_results: list,
    repeat: int,
    is_root_rank: bool,
    is_mp: bool,
    multiprocess_runtime_context: gqe.lib.MultiProcessRuntimeContext,
    validate_results: bool,
    validate_dir: bool,
    suite_name: str,
    quiet: bool,
    pipe: subprocessing.Pipe,
):
    if is_root_rank:
        data_info_id = edb.insert_data_info(upcast_to_super(data, exp.DataInfo))

        data.data_info_id = data_info_id
        data_info_ext_id = edb.insert_gqe_data_info_ext(upcast_to_super(data, GqeDataInfoExt))
    debug_mem_usage = bool(os.getenv("GQE_PYTHON_DEBUG_MEM_USAGE", False))

    task_manager_params = optimization_parameters.from_catalog_context(cat_ctx)

    load_all_data = cat_ctx.load_data_of_query == 0
    catalog = None
    if load_all_data:
        print_mp(
            f"Attempting to load full {suite_name} dataset into memory",
            is_root_rank and not quiet,
        )
        try:
            # Create the task manager context
            if is_mp:
                # FIXME: Scheduler type needs to be reworked to use query context
                # instead of task manager context. It's currently set to ALL_TO_ALL for
                # data loading, but we use ROUND_ROBIN for query execution.
                context = MultiProcessContext(
                    multiprocess_runtime_context,
                    task_manager_params,
                    gqe.lib.scheduler_type.ALL_TO_ALL,
                )
            else:
                context = Context(
                    task_manager_params,
                    debug_mem_usage=debug_mem_usage,
                    cupti_metrics=cupti_metrics,
                )
            catalog = Catalog(context)
            table_definitions = catalog.register_tables(
                **asdict(cat_ctx),
            )
        except Exception as error:
            print(f"Error registering table: {error}", is_root_rank)
            print(
                f"load_all_data failed to load due to context creation or table registration, discarding remaining {len(parameters)} experiments",
            )
            pipe_send(pipe, False)
            errors.append(("load_all_data", f"{error}"))
            parameters[:] = []
            return
        # Typically, we would expect a pipe_send(True) here, but it will be satisfied by the per-query send below.

    previous_query_str = None
    while parameters:
        # pop lets main process know we are making forward progress
        parameter = parameters.pop(0)
        query_info_ctx = parameter.query_info_ctx

        if not quiet:
            print_mp(
                f"Running query from {query_info_ctx.query_source} with parameters "
                f"debug_mem_usage={debug_mem_usage}, "
                f"{parameter}, {data}",
                is_root_rank,
            )
        elif previous_query_str != query_info_ctx.query_str:
            print_mp(f"Running query {query_info_ctx.query_str}", is_root_rank)
            previous_query_str = query_info_ctx.query_str

        # Custom queries wont enter this block, as load_all_data is always True, but they have query_idx = -1
        # Reload dataset if new query and not load_all
        if not load_all_data and cat_ctx.load_data_of_query != query_info_ctx.query_idx:
            # Python uses refcounting for object destruction. Objects are
            # immediately destroyed when the reference count reaches 0.  GC is
            # only required when we have reference cycles which cant be cleaned
            # up by refcounting.
            #
            # Context is explicitly destroyed after the Catalog so that GPU
            # memory is freed before table registration.  Catalog holds a
            # pointer to task_manager_ctx from Context, so it must be destroyed
            # first.
            #
            # Reference:
            # See https://docs.python.org/3.14/library/gc.html
            catalog = None
            context = None

            print_mp(
                f"Attempting to load TPCH Query {query_info_ctx.query_idx} data into memory",
                is_root_rank,
            )
            try:
                # Recreate context for new query data
                opt_params = optimization_parameters.from_catalog_context(cat_ctx)
                if is_mp:
                    context = MultiProcessContext(
                        multiprocess_runtime_context,
                        opt_params,
                        gqe.lib.scheduler_type.ALL_TO_ALL,
                    )
                else:
                    context = Context(
                        opt_params,
                        debug_mem_usage=debug_mem_usage,
                        cupti_metrics=cupti_metrics,
                    )
                catalog = Catalog(context)
                # Set up context with new query ID
                cat_ctx.load_data_of_query = query_info_ctx.query_idx
                table_definitions = catalog.register_tables(
                    **asdict(cat_ctx),
                )
            except Exception as error:
                print(
                    f"Error creating context or registering in memory table for query {query_info_ctx.query_idx} {type(error).__name__}: {error}",
                )
                pipe_send(pipe, False)
                # Query is not built yet without table definitions so just build the Q identifier here.
                errors.append((f"Q{query_info_ctx.query_str} load_query_data", f"{error}"))
                # Since we failed to load this data set, purge the remaining matching queries.
                parameters[:] = list(
                    filter(
                        lambda p: p.query_info_ctx.query_idx != query_info_ctx.query_idx,
                        list(parameters),
                    )
                )
                # We can continue processing since this is one query; on next iter we reload data
                # if the error corrupts the cuda context though, we need to start over
                if is_unrecoverable_error(error):
                    break
                else:
                    continue
        else:
            print_mp("No load required - data already in memory", is_root_rank and not quiet)

        # if we make it here, communicate we succeeded data load and/or didn't need to load data
        pipe_send(pipe, True)

        print_mp("Refreshing query execution context...", is_root_rank and not quiet)
        try:
            opt_params = optimization_parameters.from_query_context(parameter, data)
            # Build QueryInfo in try to catch file not found or permissions exception with substrait file
            query = _get_query_info(
                query_info_ctx,
                load_all_data,
                cat_ctx.storage_kind,
                table_definitions,
                catalog,
                multiprocess_runtime_context,
            )
            # Refresh the query context with new optimization parameters
            context.refresh_query_context(opt_params)
        except Exception as error:
            print("Error refreshing query context")
            print(f"{type(error).__name__}: {error}")
            errors.append(
                (
                    f"Q{query_info_ctx.query_str} refresh_context",
                    parameter,
                    f"{error}",
                )
            )
            pipe_send(pipe, False)
            if is_unrecoverable_error(error):
                break
            else:
                continue
        # confirm we refreshed context properly
        pipe_send(pipe, True)

        if is_root_rank:
            parameter.sut_info_id = edb_info.sut_info_id
            parameters_id = edb.insert_gqe_parameters(upcast_to_super(parameter, GqeParameters))

            # Normalize query_source: both "substrait" and "custom_substrait" are stored as "substrait"
            normalized_source = (
                "substrait"
                if "substrait" in query_info_ctx.query_source
                else query_info_ctx.query_source
            )

            query_info_id = edb.insert_query_info(
                exp.QueryInfo(
                    name=query.identifier,
                    suite=suite_name,
                    source=normalized_source,
                )
            )

            experiment_id = edb.insert_experiment(
                Experiment(
                    sut_info_id=edb_info.sut_info_id,
                    parameters_id=parameters_id,
                    hw_info_id=edb_info.hw_info_id,
                    build_info_id=edb_info.build_info_id,
                    data_info_id=data_info_id,
                    data_info_ext_id=data_info_ext_id,
                    query_info_id=query_info_id,
                    sample_size=repeat,
                )
            )

            table_stats = gqe.lib.get_table_stats(catalog._catalog)
            for table_name, stats in table_stats.items():
                # We also register parquet tables, but that doesn't undergo `in_memory_write_task` and would not have compression stats logged. Ignore here.
                if stats.num_row_groups == 0:
                    continue
                gqe_table_stats_id = edb.insert_gqe_table_stats(
                    GqeTableStats(
                        data_info_ext_id=data_info_ext_id,
                        query_info_id=query_info_id,
                        table_name=table_name,
                        stats=stats,
                    )
                )
                column_names = catalog._catalog.column_names(table_name)
                for col_idx in range(stats.num_columns):
                    if stats.column_stats[col_idx].is_string_column():
                        # Insert two entries for char and offset since they are stored in separate buffers but logically one column.
                        edb.insert_gqe_column_stats(
                            GqeColumnStats(
                                gqe_table_stats_id=gqe_table_stats_id,
                                column_name=column_names[col_idx],
                                col_idx=col_idx,
                                stats=stats,
                                column_part="char",
                            )
                        )
                        edb.insert_gqe_column_stats(
                            GqeColumnStats(
                                gqe_table_stats_id=gqe_table_stats_id,
                                column_name=column_names[col_idx],
                                col_idx=col_idx,
                                stats=stats,
                                column_part="offset",
                            )
                        )
                    else:
                        edb.insert_gqe_column_stats(
                            GqeColumnStats(
                                gqe_table_stats_id=gqe_table_stats_id,
                                column_name=column_names[col_idx],
                                col_idx=col_idx,
                                stats=stats,
                                column_part="value",
                            )
                        )

            print_mp(f"Running {query.identifier}...", is_root_rank and not quiet)

        """
        This conditional is added to modify the scheduler type to ROUND_ROBIN for query execution in multi-process mode.
        """
        if is_mp:
            multiprocess_runtime_context.update_scheduler(gqe.lib.scheduler_type.ROUND_ROBIN)

        for count in range(repeat):
            out_file = os.path.join(f"{validate_dir}", f"{query.identifier}_out.parquet")

            with nvtx.annotate(f"Run {query.identifier}"):
                try:
                    print_mp(
                        f"Starting query {query.identifier} repetition {count}...",
                        is_root_rank and not quiet,
                    )
                    duration_s, stage_durations, metric_values = context.execute(
                        catalog, query.root_relation, out_file
                    )
                except Exception as error:
                    err_str = f"{type(error).__name__}: {error}"
                    print("Error during query execution")
                    print(err_str)
                    errors.append((f"{query.identifier} query execution", parameter, f"{error}"))
                    pipe_send(pipe, False)
                    insert_gqe_run_error(edb, experiment_id, count, is_root_rank, err_str)
                    if is_unrecoverable_error(error):
                        return
                    else:
                        break
                    break
            if validate_results:
                # All ranks validate result, alternatively we need to communicate if there is an error
                try:
                    print_mp("Start validation...", is_root_rank and not quiet)
                    validate_parquet(out_file, query.reference_solution, query.validator)
                except Exception as error:
                    err_str = f"{type(error).__name__}: {error}"
                    print("Error validating solution")
                    print(err_str)
                    invalid_results.append((query.identifier, parameter))
                    pipe_send(pipe, False)
                    insert_gqe_run_error(edb, experiment_id, count, is_root_rank, err_str)
                    if is_unrecoverable_error(error):
                        return
                    else:
                        break
            else:
                print_mp("Skipping validation...", is_root_rank and not quiet)
            # Logging on host process creates a small race condition; sending before doing DB insertion
            # helps order the print messages w/o having to resort to more complicated syncronization.
            pipe_send(pipe, True)
            if is_root_rank:
                edb.insert_run(
                    exp.Run(
                        experiment_id=experiment_id,
                        number=count,
                        nvtx_marker=None,
                        duration_s=duration_s,
                    )
                )

                for stage, duration_s in stage_durations:
                    metric_id = edb.insert_metric_info(GqeMetricInfo(name=stage))
                    edb.insert_gqe_run_ext(
                        GqeRunExt(
                            experiment_id=experiment_id,
                            run_number=count,
                            metric_info_id=metric_id,
                            metric_value=duration_s,
                        )
                    )

                if cupti_metrics:
                    for metric in cupti_metrics:
                        metric_id = edb.insert_metric_info(GqeMetricInfo(name=metric))

                        edb.insert_gqe_run_ext(
                            GqeRunExt(
                                experiment_id=experiment_id,
                                run_number=count,
                                metric_info_id=metric_id,
                                metric_value=metric_values[metric],
                            )
                        )

    # Explicit cleanup in correct order to avoid segfault at exit.
    # Catalog holds a pointer to task_manager_ctx from Context, so it must be destroyed first.
    catalog = None
    context = None


def insert_gqe_run_error(edb, experiment_id, count, is_root_rank, error):
    if is_root_rank:
        edb.insert_failed_run(
            exp.FailedRun(experiment_id=experiment_id, number=count, error_msg=error)
        )
