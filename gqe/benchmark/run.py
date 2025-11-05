# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import Catalog, Context, MultiProcessContext
from gqe.benchmark.gqe_experiment import GqeExperimentConnection
import gqe.lib
from gqe.benchmark.verify import verify_parquet
from gqe.benchmark.gqe_experiment import (
    GqeParameters,
    GqeDataInfoExt,
    GqeMetricInfo,
    GqeRunExt,
)
from gqe.relation import (
    Relation,
    ReadRelation,
    FilterRelation,
    AggregateRelation,
    BroadcastJoinRelation,
    ShuffleJoinRelation,
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
from gqe.execute import MultiProcessContext, MultiProcessRuntimeContext

from database_benchmarking_tools import experiment as exp
from database_benchmarking_tools.experiment import ExperimentDB

import importlib.resources
import os
import nvtx
import pandas as pd
import re
from dataclasses import dataclass, asdict, fields
from typing import Optional
from collections.abc import Callable
import copy
import argparse

# Alias multiprocessing due to namespace class in GQE
import multiprocessing as subprocessing
import gc
import gqe.lib as lib


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
    load_all_data_from: str
    identifier_type: gqe.lib.TypeId
    use_opt_char_type: bool
    in_memory_table_compression_format: str
    in_memory_table_compression_data_type: str
    compression_chunk_size: int
    zone_map_partition_size: int
    debug_mem_usage: bool = False


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
    sut_creation_path = importlib.resources.files("gqe.benchmark").joinpath(
        "system_under_test.sql"
    )
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
        identifier_type = (
            gqe.lib.TypeId.int32 if scale_factor < 357 else gqe.lib.TypeId.int64
        )
    return identifier_type


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
    relation: Relation, query: int, fixed_read_relations: list[int] = []
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
        fix_partial_filter_column_references(
            relation.left_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.right_table, query, fixed_read_relations
        )
    # Only check instance is not enough as import_module would create different class objects.
    elif relation.__class__.__name__ == "Q10FusedProbesJoinMapBuildRelation":
        fix_partial_filter_column_references(
            relation.build_side_table, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q10FusedProbesJoinMultimapBuildRelation":
        fix_partial_filter_column_references(
            relation.build_side_table, query, fixed_read_relations
        )
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
        fix_partial_filter_column_references(
            relation.nation_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.customer_table, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q10SortLimitRelation":
        fix_partial_filter_column_references(
            relation.input_table, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q10UniqueKeyInnerJoinBuildRelation":
        fix_partial_filter_column_references(
            relation.build_side_table, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q10UniqueKeyInnerJoinProbeRelation":
        fix_partial_filter_column_references(
            relation.build_side_map, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.build_side_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.probe_side_table, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q13GroupjoinProbeRelation":
        fix_partial_filter_column_references(
            relation.groupjoin_build, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.orders, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q13FusedFilterProbeRelation":
        fix_partial_filter_column_references(
            relation.groupjoin_build, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.orders, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q16FusedFilterJoinRelation":
        fix_partial_filter_column_references(
            relation.supplier_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.part_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.partsupp_table, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q21LeftAntiJoinProbeRelation":
        fix_partial_filter_column_references(
            relation.left_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.right_table, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q21LeftSemiJoinProbeRelation":
        fix_partial_filter_column_references(
            relation.left_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.right_table, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q21LeftAntiJoinRetrieveRelation":
        fix_partial_filter_column_references(
            relation.left_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.probe, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q21LeftSemiJoinRetrieveRelation":
        fix_partial_filter_column_references(
            relation.left_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.probe, query, fixed_read_relations
        )
    elif relation.__class__.__name__ == "Q22MarkJoinRelation":
        fix_partial_filter_column_references(
            relation.customer_table, query, fixed_read_relations
        )
        fix_partial_filter_column_references(
            relation.orders_table, query, fixed_read_relations
        )
    else:
        fix_partial_filter_column_references(
            relation.input, query, fixed_read_relations
        )


def log_physical_plan(query_str: str, relation: lib.Relation, folder_path: str):
    file_path = os.path.join(folder_path, query_str + "_plan.json")
    lib.log_physical_plan(relation, file_path)


def _get_tpc_query_info(
    query_info_ctx: QueryInfoContext,
    load_all_data: bool,
    storage_kind: str,
    table_definitions: TPCHTableDefinitions,
    catalog: gqe.Catalog,
    multiprocess_runtime_context: lib.MultiProcessRuntimeContext,
):
    if query_info_ctx.query_source == "handcoded":
        query_identifier = "tpch_q" + query_info_ctx.query_str
        module = importlib.import_module(f"gqe.benchmark.{query_identifier}")
        query_object = getattr(module, query_identifier)(
            scale_factor=query_info_ctx.scale_factor
        )
        root_relation = query_object.root_relation(table_definitions)
        query = QueryInfo(
            f"Q{query_info_ctx.query_str}", root_relation, query_info_ctx.reference_file
        )
        # Fix partial filter column references for handcoded queries
        if not load_all_data and (storage_kind != "parquet_file"):
            fix_partial_filter_column_references(
                root_relation, query_info_ctx.query_idx
            )
        if validator := get_query_validator(query_object):
            query.validator = validator

        if query_info_ctx.physical_plan_folder:
            root_relation.log_physical_plan(
                f"Q{query_info_ctx.query_str}", query_info_ctx.physical_plan_folder
            )
    elif query_info_ctx.query_source == "substrait":
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


def run_tpc(
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
    multiprocess_runtime_context: lib.MultiProcessRuntimeContext,
    verify_results: bool,
    quiet: bool = False,
    pipe: subprocessing.Pipe = None,
):
    # only send DB to root rank so we will get error if there is a logic mistake
    if is_root_rank:
        gqe_host = "localhost"
        edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(
            GqeExperimentConnection
        )
        with edb_config as edb:
            _run_tpc(
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
                verify_results,
                quiet,
                pipe,
            )
    else:
        _run_tpc(
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
            verify_results,
            quiet,
            pipe,
        )


def pipe_send(pipe: subprocessing.Pipe, status: bool):
    if pipe is not None:
        pipe.send(status)


# if cuda error matches any of these regex, it's unrecoverable and we need a new process
def is_unrecoverable_error(e):
    error = f"{e}"
    messages = [
        "cudaErrorIllegalAddress",
        "cudaErrorInvalidAddressSpace",
        "cudaErrorMisalignedAddress",
    ]
    for message in messages:
        if message in error:
            return True
    return False


def _run_tpc(
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
    multiprocess_runtime_context: lib.MultiProcessRuntimeContext,
    verify_results: bool,
    quiet: bool,
    pipe: subprocessing.Pipe,
):
    if is_root_rank:
        data_info_id = edb.insert_data_info(upcast_to_super(data, exp.DataInfo))

        data.data_info_id = data_info_id
        data_info_ext_id = edb.insert_gqe_data_info_ext(
            upcast_to_super(data, GqeDataInfoExt)
        )
    debug_mem_usage = bool(os.getenv("GQE_PYTHON_DEBUG_MEM_USAGE", False))
    cat_ctx.debug_mem_usage = debug_mem_usage

    load_all_data = cat_ctx.load_data_of_query == 0
    catalog = None
    if load_all_data:
        print_mp(
            "Attempting to load full TPCH dataset into memory",
            is_root_rank and not quiet,
        )
        try:
            catalog = Catalog()
            table_definitions = catalog.register_tpch(
                **asdict(cat_ctx),
                multiprocess_runtime_context=multiprocess_runtime_context,
            )
        except Exception as error:
            print(f"Error registering table: {error}", is_root_rank)
            print(
                f"load_all_data failed to load, discarding remaining {len(parameters)} experiments",
            )
            pipe_send(pipe, False)
            errors.append(("load_all_data", error))
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
                f"{ parameter }, {data}",
                is_root_rank,
            )
        elif previous_query_str != query_info_ctx.query_str:
            print_mp(f"Running query {query_info_ctx.query_str}", is_root_rank)
            previous_query_str = query_info_ctx.query_str

        # Reload dataset if new query and not load_all
        if not load_all_data and cat_ctx.load_data_of_query != query_info_ctx.query_idx:
            # there is a potential race condition with the garbage collector on reassignment, so
            # let's just explicitly delete and try to reclaim
            del catalog
            gc.collect()
            print_mp(
                f"Attempting to load TPCH Query {query_info_ctx.query_idx} data into memory",
                is_root_rank,
            )
            try:
                catalog = Catalog()
                # Set up context with new query ID
                cat_ctx.load_data_of_query = query_info_ctx.query_idx
                table_definitions = catalog.register_tpch(
                    **asdict(cat_ctx),
                    multiprocess_runtime_context=multiprocess_runtime_context,
                )
            except Exception as error:
                print(
                    f"Error registering in memory table for query {query_info_ctx.query_idx} {type(error).__name__}: {error}",
                )
                pipe_send(pipe, False)
                # Query is not built yet without table definitions so just build the Q identifier here.
                errors.append((f"Q{query_info_ctx.query_str} load_query_data", error))
                # Since we failed to load this data set, purge the remaining matching queries.
                parameters[:] = list(
                    filter(
                        lambda p: p.query_info_ctx.query_idx
                        != query_info_ctx.query_idx,
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
            print_mp(
                "No load required - data already in memory", is_root_rank and not quiet
            )
        # if we make it here, communicate we succeeded data load and/or didn't need to load data
        pipe_send(pipe, True)

        # need to load query info regardless of which catalog path we take
        query = _get_tpc_query_info(
            query_info_ctx,
            load_all_data,
            cat_ctx.storage_kind,
            table_definitions,
            catalog,
            multiprocess_runtime_context,
        )

        context_params = (
            parameter.num_workers,
            parameter.num_partitions,
            data.char_type == "char",
            parameter.use_overlap_mtx,
            parameter.join_use_hash_map_cache,
            parameter.read_use_zero_copy,
            parameter.join_use_unique_keys,
            parameter.join_use_perfect_hash,
            parameter.join_use_mark_join,
            data.compression_format,
            data.compression_data_type,
            data.compression_chunk_size,
            parameter.use_partition_pruning,
            data.zone_map_partition_size,
            parameter.filter_use_like_shift_and,
            parameter.aggregation_use_perfect_hash,
        )
        print_mp("Building query execution context...", is_root_rank and not quiet)
        try:
            if is_mp:
                context = MultiProcessContext(
                    multiprocess_runtime_context,
                    *context_params,
                    lib.scheduler_type.ROUND_ROBIN,
                )
            else:
                context = Context(
                    *context_params,
                    debug_mem_usage=debug_mem_usage,
                    cupti_metrics=cupti_metrics,
                )
        except Exception as error:
            print("Error constructing query context")
            print(f"{type(error).__name__}: {error}")
            errors.append((f"{query.identifier} construct_context", parameter, error))
            pipe_send(pipe, False)
            if is_unrecoverable_error(error):
                break
            else:
                continue
        # confirm we loaded context properly
        pipe_send(pipe, True)

        if is_root_rank:
            parameter.sut_info_id = edb_info.sut_info_id
            parameters_id = edb.insert_gqe_parameters(
                upcast_to_super(parameter, GqeParameters)
            )

            experiment_id = edb.insert_experiment(
                Experiment(
                    sut_info_id=edb_info.sut_info_id,
                    parameters_id=parameters_id,
                    hw_info_id=edb_info.hw_info_id,
                    build_info_id=edb_info.build_info_id,
                    data_info_id=data_info_id,
                    data_info_ext_id=data_info_ext_id,
                    name=query.identifier,
                    suite="TPC-H",
                    scale_factor=scale_factor,
                    query_source=query_info_ctx.query_source,
                )
            )
            print_mp(f"Running {query.identifier}...", is_root_rank and not quiet)

        for count in range(repeat):
            out_file = f"{query.identifier}_out.parquet"

            with nvtx.annotate(f"Run {query.identifier}"):
                try:
                    print_mp(
                        f"Starting query {query.identifier} repetition {count}...",
                        is_root_rank and not quiet,
                    )
                    duration_s, metric_values = context.execute(
                        catalog, query.root_relation, out_file
                    )
                except Exception as error:
                    print("Error during query execution")
                    print(f"{type(error).__name__}: {error}")
                    errors.append(
                        (f"{query.identifier} query execution", parameter, error)
                    )
                    pipe_send(pipe, False)
                    break
            if verify_results:
                # All ranks verify result, alternatively we need to communicate if there is an error
                try:
                    print_mp("Start verification...", is_root_rank and not quiet)
                    verify_parquet(out_file, query.reference_solution, query.validator)
                except Exception as error:
                    print("Error verifying solution")
                    print(f"{type(error).__name__}: {error}")
                    invalid_results.append((query.identifier, parameter))
                    pipe_send(pipe, False)
                    success = False
                    if is_unrecoverable_error(error):
                        return
                    else:
                        break
            else:
                print_mp("Skipping verification...", is_root_rank and not quiet)
            # Logging on host process creates a small race condition; sending before doing DB insertion
            # helps order the print messages w/o having to resort to more complicated syncronization.
            pipe_send(pipe, True)
            if is_root_rank:
                run_id = edb.insert_run(
                    exp.Run(
                        experiment_id=experiment_id,
                        number=count,
                        nvtx_marker=None,
                        duration_s=duration_s,
                    )
                )

                if cupti_metrics:
                    for metric in cupti_metrics:
                        metric_id = edb.insert_metric_info(GqeMetricInfo(name=metric))

                        edb.insert_gqe_run_ext(
                            GqeRunExt(
                                run_id=run_id,
                                metric_info_id=metric_id,
                                metric_value=metric_values[metric],
                            )
                        )

        del context
        gc.collect()
