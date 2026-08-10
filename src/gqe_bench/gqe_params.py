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

"""Parameter generation for sweep and pretuned modes."""

import itertools
import logging
import re
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

from gqe_bench.arguments import get_query_execution_params
from gqe_bench.query_source import QuerySource
from gqe_bench.schema import (
    DATA_INFO_MAPPING,
    Q_NAME,
    Q_SOURCE,
    QUERY_PARAMS_MAPPING,
    SCALE_FACTOR,
    DataInfo,
    DataLoadGroup,
    Query,
    QueryParams,
    query_best_parameters,
)
from gqe_bench.suites import get_suite

logger = logging.getLogger(__name__)


def _warn_field_dependencies(args: Namespace) -> None:
    """Warn about sweep fields that only take effect with a companion field."""
    compression_level = args.compression_level
    use_cpu = args.use_cpu_compression
    default_level = DATA_INFO_MAPPING.field_default("compression_level")

    if (
        compression_level is not None
        and compression_level != [default_level]
        and (use_cpu is None or True not in use_cpu)
    ):
        logger.warning(
            "compression_level=%s has no effect without use_cpu_compression=True",
            compression_level,
        )

    zone_map = args.zone_map_partition_size
    pruning = args.use_partition_pruning

    if zone_map is not None and (pruning is None or True not in pruning):
        logger.warning(
            "zone_map_partition_size=%s has no effect without use_partition_pruning=True",
            zone_map,
        )
    if pruning is not None and True in pruning and zone_map is None:
        logger.warning(
            "use_partition_pruning=True requires zone_map_partition_size to be set",
        )

    storage = args.storage_device_kind
    if storage is not None and any(k != "boost_shared_memory" for k in storage):
        logger.warning(
            "storage_device_kind=%s: gqe does not currently support setting "
            "the device kind over the CLI; only boost_shared_memory is honored.",
            storage,
        )


def _prune(query: Query, qp: QueryParams, di: DataInfo) -> bool:
    """Return True if this (Query, QueryParams) combination should be skipped."""
    if qp.read_use_zero_copy and qp.num_partitions != di.num_row_groups:
        return True
    if qp.read_use_zero_copy and di.storage_device_kind == "parquet_file":
        return True
    if qp.read_use_zero_copy and di.compression_format != "none":
        return True
    if qp.num_workers > qp.num_partitions:
        return True
    if di.compression_format != "none" and qp.use_overlap_mtx:
        return True
    # Perfect hash doesn't support hash map cache; one must be on, other off (gqe#161)
    # Performance is extremely bad with neither enabled
    if qp.join_use_perfect_hash == qp.join_use_hash_map_cache:
        return True
    # Perfect hash unsupported in SQL pipeline; handcoded sets it per-operator (gqe#161)
    if query.source == QuerySource.SQL and (
        qp.join_use_perfect_hash or qp.aggregation_use_perfect_hash
    ):
        return True
    # Large SF + few partitions hits cuDF type limits; Q11/Q20 exempt (gqe#221)
    if di.scale_factor > 500 and qp.num_partitions < 4:
        base_num = re.match(r"q?(\d+)", query.name)
        if base_num and int(base_num.group(1)) not in (11, 20):
            return True
    # scatter_aggregate has no fixed-point output path, so perfect-hash
    # aggregation throws on decimal columns. Drop the combination until the
    # engine supports it.
    if di.decimal_type == "decimal" and qp.aggregation_use_perfect_hash:
        return True
    return False


def generate_sweep_groups(args: Namespace, schema_ddl: str) -> list[DataLoadGroup]:
    """Generate DataLoadGroups from sweep config via Cartesian product."""
    _warn_field_dependencies(args)
    suite = get_suite(args.suite_name)

    # Validate query sources before parsing the DDL — keeps the failure mode
    # of an unsupported suite/source combination as ValueError, not whatever
    # downstream parse error the DDL might raise.
    query_sources = args.query_source
    for qs in query_sources:
        try:
            suite.available_queries(qs, None)
        except (ValueError, NotImplementedError):
            raise ValueError(f"Suite '{args.suite_name}' does not support query source '{qs}'")

    dataset_props = suite.read_dataset_properties(schema_ddl)
    dataset_props["scale_factor"] = suite.infer_scale_factor(args.dataset)
    dataset_props["location"] = str(args.dataset)  # dataset path as DataInfo key

    # Query identities only; the payload is built per data configuration below.
    query_ids = [
        (qs, name) for qs in query_sources for name in suite.available_queries(qs, args.queries)
    ]
    if not query_ids:
        raise ValueError(f"No queries found for suite '{args.suite_name}'")

    # A query's payload depends on the decimal representation, which is a DataInfo
    # field, so it is resolved where the query meets its DataInfo. Keyed by query so
    # one unproducible query reports once rather than once per data configuration.
    problems: dict[tuple[QuerySource, str], str] = {}

    data_sweep_lists = [getattr(args, f) for f in DATA_INFO_MAPPING.sweep_fields]

    # Grouping key: (DataInfo, None) bundles all queries of a DataInfo into one
    # group (load_all_data=true). (DataInfo, base_query_name) emits one group per
    # (DataInfo, base query), so a base query and its variants (e.g. 2 and
    # 2_fused_filter) load together once instead of reloading identical data.
    groups: dict[tuple[DataInfo, str | None], list[tuple[Query, QueryParams]]] = defaultdict(list)

    # Three levels: one data configuration, times every query, times every
    # execution-parameter combination for that query.
    for data_combo in itertools.product(*data_sweep_lists):
        data_kwargs = dict(zip(DATA_INFO_MAPPING.sweep_fields, data_combo))
        data_kwargs.update(dataset_props)
        data_info = DataInfo(**data_kwargs)

        for query_source, name in query_ids:
            # Resolved here rather than up front because the payload depends on
            # data_info.decimal_type: the plan's read types and literals differ
            # between the float and fixed-point representations.
            try:
                content = suite.resolve_content(
                    name,
                    query_source,
                    suite.query_file(name, args.sql),
                    dataset_props.get("scale_factor"),
                    dataset_props["identifier_type"],
                    args.load_all_data,
                    data_info.decimal_type,
                )
            except ValueError as e:
                problems[(query_source, name)] = f"{query_source.value} query '{name}': {e}"
                continue
            query = Query(
                name=name,
                source=query_source,
                reference_file=suite.solution_file(name, args.solution),
                content=content,
            )

            qp_overrides = get_query_execution_params(args, query.name)
            query_sweep_lists = [qp_overrides[f] for f in QUERY_PARAMS_MAPPING.sweep_fields]

            # Per-query overrides replace the global sweep lists for matched queries.
            for query_combo in itertools.product(*query_sweep_lists):
                qp = QueryParams(**dict(zip(QUERY_PARAMS_MAPPING.sweep_fields, query_combo)))
                if not _prune(query, qp, data_info):
                    key = (
                        data_info,
                        None if args.load_all_data else suite.base_query_name(query.name),
                    )
                    groups[key].append((query, qp))

    if problems:
        raise ValueError("Cannot produce queries:\n  " + "\n  ".join(problems.values()))

    return [DataLoadGroup(data_info=di, queries=list(pairs)) for (di, _), pairs in groups.items()]


def generate_pretuned_groups(args: Namespace) -> list[DataLoadGroup]:
    """Generate DataLoadGroups from prior sweep results."""
    swept = Path(args.swept_sqlite)
    db_paths = sorted(swept.glob("*.db3")) if swept.is_dir() else [swept]
    best_params = query_best_parameters(db_paths)

    if not best_params:
        raise ValueError(f"No best parameters found in {args.swept_sqlite}")

    suite = get_suite(args.suite_name)
    solution_dir = args.solution

    # See generate_sweep_groups for the keying scheme.
    groups: dict[tuple[DataInfo, str | None], list[tuple[Query, QueryParams]]] = defaultdict(list)

    query_filter = args.queries
    source_filter = args.query_source

    for row in best_params:
        q_name = row[Q_NAME]
        if query_filter and q_name not in query_filter:
            continue
        query_source = QuerySource.from_db(row.get(Q_SOURCE, QuerySource.SQL))
        if source_filter and query_source not in source_filter:
            continue
        data_kwargs = DATA_INFO_MAPPING.extract(row)
        data_kwargs["location"] = str(args.dataset)  # dataset path as DataInfo key
        # FLOAT end-to-end so fractional SFs (0.01, 0.1) survive the round-trip.
        data_kwargs["scale_factor"] = row.get(SCALE_FACTOR) or suite.infer_scale_factor(
            args.dataset
        )
        data_info = DataInfo(**data_kwargs)

        content = suite.resolve_content(
            q_name,
            query_source,
            suite.query_file(q_name, args.sql),
            data_info.scale_factor,
            data_info.identifier_type,
            args.load_all_data,
            data_info.decimal_type,
        )
        query = Query(
            name=q_name,
            source=query_source,
            reference_file=suite.solution_file(q_name, solution_dir),
            content=content,
        )

        qp = QueryParams(**QUERY_PARAMS_MAPPING.extract(row))
        key = (data_info, None if args.load_all_data else query.name)
        groups[key].append((query, qp))

    return [DataLoadGroup(data_info=di, queries=list(pairs)) for (di, _), pairs in groups.items()]


def generate_groups(args: Namespace, schema_ddl: str) -> list[DataLoadGroup]:
    """Generate DataLoadGroups — dispatches to sweep or pretuned.

    ``schema_ddl`` is consumed by the sweep path (``generate_sweep_groups``);
    the pretuned path reads dataset properties from the prior sweep DB and
    ignores it.
    """
    if args.swept_sqlite:
        return generate_pretuned_groups(args)
    return generate_sweep_groups(args, schema_ddl)
