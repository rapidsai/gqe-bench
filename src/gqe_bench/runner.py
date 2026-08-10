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

"""Benchmark runner — orchestrates the session, data loading, and query execution."""

import logging
import os
import sys
import tempfile
from pathlib import Path

from gqe_bench.arguments import parse_args
from gqe_bench.gqe_params import generate_groups
from gqe_bench.logger import configure_logging
from gqe_bench.recording import (
    SweepContext,
    insert_data_dimensions,
    record_experiment,
    replace_run_with_failure,
    setup_db,
)
from gqe_bench.resources import ArtifactMissing
from gqe_bench.schema import DataLoadGroup
from gqe_bench.session import LOAD_ALL_COLUMNS, GqeSession, QueryFailed, RestartRequired
from gqe_bench.suites import get_suite
from gqe_bench.suites.base import Suite
from gqe_bench.validate import ValidationFailed, validate_parquet

logger = logging.getLogger(__name__)


def _run_group(
    session: GqeSession,
    group: DataLoadGroup,
    suite: type[Suite],
    sweep: SweepContext,
) -> None:
    """Execute every (Query, QueryParams) pair in ``group``.

    Pops each pair before execution. Creates a per-run tempfile for the
    CLI's parquet output and validates against ``query.reference_file``
    when set. Propagates ``RestartRequired`` so the caller can tear down
    the session. Remaining pairs stay in ``group.queries``. Other
    exceptions propagate to the outer handler.
    """
    with sweep.db_mgr as edb:
        dims = insert_data_dimensions(edb, group.data_info)

        while group.queries:
            query, qp = group.queries.pop(0)
            logger.info("Running %s...", query.name)
            logger.info("Parameters: %s", qp)
            exp_id = record_experiment(edb, dims, query, qp, sweep)
            edb.commit()

            for run_num in range(sweep.repeat):
                logger.info("Starting %s repetition %d...", query.name, run_num)
                try:
                    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
                        output_path = Path(tmp.name)
                        session.execute_query(query, output_path, exp_id, qp)
                        if query.reference_file is not None:
                            logger.info("Validating result...")
                            validate_parquet(output_path, query.reference_file, query.name)
                    logger.info("Run %d: ok", run_num)
                except ValidationFailed as e:
                    logger.warning("Run %d: validation failed: %s", run_num, e)
                    replace_run_with_failure(edb, exp_id, run_num, str(e))
                    edb.commit()
                    break
                except RestartRequired as e:
                    logger.warning("Run %d: restart required: %s", run_num, e)
                    replace_run_with_failure(edb, exp_id, run_num, str(e))
                    edb.commit()
                    raise
                except QueryFailed as e:
                    logger.warning("Run %d: query failed: %s", run_num, e)
                    replace_run_with_failure(edb, exp_id, run_num, str(e))
                    edb.commit()
                    break

            logger.info("Parameter sets remaining: %d", len(group.queries))


def _verify_reference_files(groups: list[DataLoadGroup]) -> None:
    """Fail before server start if any query's reference parquet is missing or empty.

    Reference paths are resolved at group-generation time, before the server,
    data load, or any query runs. A missing or 0-byte file is an environment
    error (0 bytes is never valid parquet); surface it here rather than
    crashing mid-sweep when validate_parquet first reads it. Queries without a
    reference (no --solution) are skipped.
    """
    refs = {
        q.reference_file
        for group in groups
        for q, _ in group.queries
        if q.reference_file is not None
    }
    problems: list[str] = []
    for ref in sorted(refs, key=str):
        if not ref.is_file():
            problems.append(f"{ref} (missing)")
        elif ref.stat().st_size == 0:
            problems.append(f"{ref} (empty)")
    if problems:
        raise FileNotFoundError(
            "Reference parquet file(s) missing or empty:\n  " + "\n  ".join(problems)
        )


def main() -> None:
    """Entry point: parse args, set up the experiment DB, and run the sweep across groups with server-restart recovery."""
    args = parse_args()
    configure_logging(args)

    suite = get_suite(args.suite_name)
    schema_ddl = suite.schema(args.schema)
    tables = suite.tables(args.dataset)
    if not tables:
        raise FileNotFoundError(f"No table directories found in {args.dataset}")

    groups = generate_groups(args, schema_ddl)
    total_pairs = sum(len(g.queries) for g in groups)
    logger.info("Generated %d data configs, %d total parameter sets", len(groups), total_pairs)
    _verify_reference_files(groups)

    try:
        # Wraps the session so teardown runs LIFO: the engine stops first, then
        # the database closes with no other writer still holding the file.
        with setup_db(args) as sweep:
            env = os.environ | sweep.env

            while groups:
                try:
                    with GqeSession(args, env, schema_ddl, tables) as session:
                        while groups:
                            group = groups[0]
                            logger.info(
                                "=== Data config (%d remaining) ===\n  %s",
                                len(groups),
                                group.data_info,
                            )

                            try:
                                # load_all_data registers every table with every
                                # column; otherwise project to the columns this
                                # group's queries read (narrowed schema + skipped
                                # unreferenced tables).
                                if args.load_all_data:
                                    required = LOAD_ALL_COLUMNS
                                    logger.info("Loading all %d tables", len(tables))
                                else:
                                    # Distinct queries only; keyed on Query since names repeat across sources.
                                    unique_queries = list(
                                        dict.fromkeys(q for q, _ in group.queries)
                                    )
                                    required = suite.required_columns(unique_queries, args.schema)
                                    logger.info(
                                        "Loading data: %d of %d tables referenced",
                                        len(required),
                                        len(tables),
                                    )
                                session.load_data(required, group.data_info)
                            except RestartRequired:
                                groups.pop(0)
                                raise

                            _run_group(session, group, suite, sweep)
                            groups.pop(0)

                except RestartRequired as e:
                    logger.warning("Restart required (%s); restarting server...", e)

            logger.info("Finished SQLite file at %s", args.output)
    except ArtifactMissing as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
