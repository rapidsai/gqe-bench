#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe.benchmark.gqe_experiment import GqeExperimentConnection
from gqe.benchmark.run import (
    CatalogContext,
    QueryInfoContext,
    run_tpc,
    DataInfo,
    QueryInfo,
    QueryExecutionContext,
    setup_db,
    parse_scale_factor,
    set_eager_module_loading,
    parse_bool,
    parse_identifier_type,
    is_valid_identifier_type,
    fix_partial_filter_column_references,
    get_query_validator,
    print_mp,
    boost_shared_memory_pool_size,
)
from gqe import lib


from database_benchmarking_tools.experiment import ExperimentDB
from database_benchmarking_tools.utility import generate_db_path

import argparse
import importlib
import itertools

# We rename this because of namespace clash with multiprocessing in GQE
import multiprocessing as subprocessing
import signal
import sys
import time
import os


def get_queries(query_source: str, queries: list[str] = None):
    handcoded_queries = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "9",
        "10",
        "11",
        "12",
        "13",
        "13_opt",
        "13_fused",
        "15",
        "16",
        "16_opt",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "22_opt",
    ]

    # 1, 15, 18, 21 removed as per previous run-script
    # Q11 FIXME: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/141
    # Q4 removed as the substrait plan fails for SF1k
    substrait_queries = [
        "2",
        "3",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "12",
        "13",
        "14",
        "16",
        "17",
        "19",
        "20",
        "22",
    ]

    if queries:
        handcoded_queries = sorted(set(handcoded_queries) & set(queries))
        substrait_queries = sorted(set(substrait_queries) & set(queries))

    if query_source == "handcoded":
        if len(handcoded_queries) == 0:
            raise ValueError(f"No handcoded queries found for the given queries")
        substrait_queries = []
    elif query_source == "substrait":
        if len(substrait_queries) == 0:
            raise ValueError(f"No substrait queries found for the given queries")
        handcoded_queries = []
    elif query_source == "both":
        pass
    else:
        raise ValueError(f"Invalid query source: {query_source}")

    return handcoded_queries, substrait_queries


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", help="TPC-H dataset location")
    arg_parser.add_argument("plan", help="Substrait query plan location")
    arg_parser.add_argument("solution", help="Reference results location with pattern")
    arg_parser.add_argument("--output", "-o", help="Output file path")
    arg_parser.add_argument(
        "--queries",
        "-q",
        help="Which queries to run",
        nargs="+",
        action="extend",
        type=str,
    )
    arg_parser.add_argument(
        "--identifier-type",
        "-i",
        help="Identifier type used in the dataset. By default, a suitable type is automatically selected.",
        choices=["int32", "int64"],
        nargs="+",
        action="extend",
        type=str,
    )
    arg_parser.add_argument(
        "--row-groups",
        "-r",
        help="Number of row groups to use",
        nargs="+",
        action="extend",
        type=int,
    )
    arg_parser.add_argument(
        "--partitions",
        "-p",
        help="Number of partitions to use",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )
    arg_parser.add_argument(
        "--workers",
        "-w",
        help="Number of workers to use",
        nargs="+",
        type=int,
        default=[1],
    )
    arg_parser.add_argument(
        "--query-source",
        help="Query source",
        choices=["handcoded", "substrait", "both"],
        default="both",
    )
    arg_parser.add_argument(
        "--compression-format",
        "-c",
        help="Compression format to use",
        choices=[
            "none",
            "ans",
            "lz4",
            "snappy",
            "gdeflate",
            "deflate",
            "cascaded",
            "zstd",
            "gzip",
            "bitcomp",
            "best_compression_ratio",
            "best_decompression_speed",
        ],
        nargs="+",
        default=["none"],
    )
    arg_parser.add_argument(
        "--load-all-data",
        "-l",
        help="Whether to load all data at once (1) or per query (0). If not specified, defaults to 1 for scale_factor <= 200, 0 otherwise",
        type=int,
        choices=[0, 1],
    )
    arg_parser.add_argument(
        "--partition-pruning",
        "-spp",
        help="Enable partition pruning optimization",
        type=parse_bool,
        nargs="*",
        default=[False],
    )
    arg_parser.add_argument(
        "--storage-kind",
        "-k",
        help="Storage kind",
        choices=[
            "pinned_memory",
            "system_memory",
            "numa_memory",
            "device_memory",
            "managed_memory",
            "numa_pinned_memory",
            "boost_shared_memory",
            "parquet_file",
        ],
        nargs="+",
        default=["numa_pinned_memory"],
    )
    arg_parser.add_argument(
        "--multiprocess", "-m", help="Run in multiprocess mode", action="store_true"
    )
    arg_parser.add_argument(
        "--repeat", "-rep", help="How many times to run each query", type=int, default=6
    )
    arg_parser.add_argument(
        "--query-timeout",
        "-qt",
        help="Timeout in (s) for queries in subprocesses. Used to kill suspected hanging jobs after timeout exceeded. Ignored if -sb is not set. Ignored if -m is set. Default: 30m",
        type=int,
        default=1800,
    )
    arg_parser.add_argument(
        "--data-timeout",
        "-dt",
        help="Timeout in (s) for data load subprocesses. Used to kill suspected hanging jobs after timeout exceeded. Ignored if -sb is not set. Ignored if -m is set. Default: 3hr",
        type=int,
        default=10800,
    )
    arg_parser.add_argument(
        "--boost_pool_size", help="Boost pool size in GBs", type=int, default=None
    )
    arg_parser.add_argument(
        "--sandboxing",
        "-sb",
        help="Run with sandboxing. Ignored if -m is set.",
        action="store_true",
    )
    arg_parser.add_argument(
        "--verify-results",
        help="Verify results before writing the timing entries to the database. Defaults to True.",
        type=parse_bool,
        default=True,
    )
    args = arg_parser.parse_args()

    if args.query_timeout < 0 or args.data_timeout < 0:
        print(f"Timeouts cannot be negative: {query_timeout}, {data_timeout}")
        print("Timeout must be a positive integer or 0")
        print("Exiting with error")
        sys.exit(1)

    if args.sandboxing and args.multiprocess:
        print(
            f"Multi-process sandboxing enabled by -sb is not compatible with multi-gpu set by -m at this time. Ignoring sandboxing."
        )

    if args.multiprocess:
        if args.storage_kind != ["parquet_file"]:
            raise ValueError(
                "Multiprocess mode is only supported with parquet_file storage kind"
            )

    gqe_host = "localhost"
    scale_factor = parse_scale_factor(args.dataset)
    load_all_data = (
        args.load_all_data
        if args.load_all_data is not None
        else (1 if scale_factor <= 200 else 0)
    )

    multiprocess_runtime_context = None
    if args.multiprocess:
        if args.storage_kind != ["parquet_file"] and args.storage_kind != [
            "boost_shared_memory"
        ]:
            raise ValueError(
                "Multiprocess mode is only supported with parquet_file storage kind or boost_shared_memory storage kind"
            )

        lib.mpi_init()

        if args.storage_kind == ["boost_shared_memory"]:
            pool_size = boost_shared_memory_pool_size(
                args.boost_pool_size, scale_factor, load_all_data
            )
            print(f"Initializing CPU shared memory with pool size {pool_size}")
            lib.initialize_shared_memory(pool_size)

        multiprocess_runtime_context = lib.MultiProcessRuntimeContext(
            lib.scheduler_type.ROUND_ROBIN, args.storage_kind[0]
        )

    if not args.identifier_type:
        identifier_type = [parse_identifier_type(args.dataset)]
    else:
        # You can set it to int32 or int64, for SF1k int64 is required.
        str_to_type = {"int32": lib.TypeId.int32, "int64": lib.TypeId.int64}
        identifier_type = [str_to_type[t] for t in args.identifier_type]

    set_eager_module_loading()

    edb_file = None
    edb_config = None
    edb_info = None
    is_root_rank = (not args.multiprocess) or (lib.mpi_rank() == 0)
    if is_root_rank:
        edb_file = (
            args.output if args.output else generate_db_path(f"gqe", "tpch", gqe_host)
        )

        edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(
            GqeExperimentConnection
        )
        edb_config.create_experiment_db()
        edb_info = None
        with edb_config as edb:
            edb_info = setup_db(edb)
        print(f"Writing SQLite file to {edb_file}")

    handcoded_queries, substrait_queries = get_queries(args.query_source, args.queries)

    def run_sweep(edb_file, edb_info, args):
        nonlocal identifier_type
        errors = []
        num_row_group_list = []
        if args.row_groups:
            num_row_group_list = args.row_groups
        elif scale_factor < 500:
            num_row_group_list = [1, 8]
        elif scale_factor < 1000:
            num_row_group_list = [4, 8]
        else:
            num_row_group_list = [16, 32]

        all_errors = []
        for (
            num_row_groups,
            use_opt_type_for_single_char_col,
            compression_format,
            compression_data_type,
            compression_chunk_size,
            identifier_type,
            storage_kind,
            zone_map_partition_size,
        ) in itertools.product(
            num_row_group_list,
            [True],
            args.compression_format,
            ["char"],
            [2**16],
            identifier_type,
            args.storage_kind,
            [100000],
        ):
            match is_valid_identifier_type(identifier_type, "tpch", scale_factor):
                case True:
                    pass
                case False:
                    continue
                case None:
                    raise ValueError(
                        f"Unknown if identifier type { identifier_type } is valid for the given dataset"
                    )

            data_info = DataInfo(
                storage_device_kind=storage_kind,
                format="internal",
                location=None,  # FIXME: set location as NUMA node, iff set in GQE
                not_null=False,
                identifier_type=str(identifier_type),
                char_type="char" if use_opt_type_for_single_char_col else "text",
                decimal_type="float",
                num_row_groups=num_row_groups,
                compression_format=compression_format,
                compression_data_type=compression_data_type,
                compression_chunk_size=compression_chunk_size,
                zone_map_partition_size=zone_map_partition_size,
            )

            if load_all_data or (storage_kind == "parquet_file"):
                cat_ctx = CatalogContext(
                    args.dataset,
                    storage_kind,
                    num_row_groups,
                    0,
                    "required",  # Load data required by 22 - TPC-H queries
                    identifier_type,
                    use_opt_type_for_single_char_col,
                    compression_format,
                    compression_data_type,
                    compression_chunk_size,
                    zone_map_partition_size,
                )
            else:
                # Pass in non-zero value to indicate load_all_data=False
                cat_ctx = CatalogContext(
                    args.dataset,
                    storage_kind,
                    num_row_groups,
                    -1,
                    "required",  # Load data required by 22 - TPC-H queries
                    identifier_type,
                    use_opt_type_for_single_char_col,
                    compression_format,
                    compression_data_type,
                    compression_chunk_size,
                    zone_map_partition_size,
                )

            # We use a regular list to build so that we can sort it later before
            # moving the data into the list proxy.
            parameters = []
            for query_source, queries in [
                ("handcoded", handcoded_queries),
                ("substrait", substrait_queries),
            ]:
                for query_str in queries:
                    query_idx = int(query_str.split("_")[0])
                    reference_file = args.solution.replace("%d", f"q{query_idx}")
                    physical_plan_folder = None
                    if query_source == "handcoded":
                        substrait_file = None
                    elif query_source == "substrait":
                        substrait_file = os.path.join(args.plan, f"df_q{query_idx}.bin")
                    query_info_ctx = QueryInfoContext(
                        query_idx,
                        query_str,
                        query_source,
                        reference_file,
                        scale_factor,
                        substrait_file,
                        physical_plan_folder,
                    )

                    for (
                        num_workers,
                        num_partitions,
                        use_overlap_mtx,
                        join_use_hash_map_cache,
                        read_use_zero_copy,
                        join_use_unique_keys,
                        join_use_perfect_hash,
                        join_use_mark_join,
                        use_partition_pruning,
                        filter_use_like_shift_and,
                        aggregation_use_perfect_hash,
                    ) in itertools.product(
                        args.workers,
                        args.partitions,
                        [False, True],
                        [False, True],
                        [False, True],
                        [True],
                        [False, True],
                        [False, True],
                        args.partition_pruning,
                        [False, True],
                        [False, True],
                    ):
                        # Skip zero copy for partition-row-group combinations where zero copy is not supported.
                        if read_use_zero_copy and (num_partitions != num_row_groups):
                            print_mp(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, num_partitions: {num_partitions}, num_row_groups: {num_row_groups} because zero copy requires row groups to be equal to partitions",
                                is_root_rank,
                            )
                            continue

                        # Zero copy can only be used for in-memory tables.
                        if read_use_zero_copy and storage_kind == "parquet_file":
                            print_mp(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, storage_kind: {storage_kind} because zero copy is not supported for parquet files",
                                is_root_rank,
                            )
                            continue

                        # Skip zero copy for compression, as compression takes precedence over zero copy in GQE read task.
                        #
                        # Note: Revisit if GQE behavior changes in future.
                        if read_use_zero_copy and compression_format != "none":
                            print_mp(
                                f"Skipping read_use_zero_copy: {read_use_zero_copy}, compression_format: {compression_format} because zero copy is not supported with compression",
                                is_root_rank,
                            )
                            continue

                        if num_workers > num_partitions:
                            print_mp(
                                f"Skipping num_workers: {num_workers}, num_partitions: {num_partitions} because num_workers greater than num_partitions",
                                is_root_rank,
                            )
                            continue

                        if compression_format != "none" and use_overlap_mtx:
                            print_mp(
                                f"Skipping compression_format: {compression_format}, use_overlap_mtx: {use_overlap_mtx} because its not optimal to use overlap matrix with compression",
                                is_root_rank,
                            )
                            continue

                        # We want to always use hash map caching except when perfect hashing is used
                        # Because perfect join doesn't support hash map cache, see: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/161
                        if join_use_perfect_hash == join_use_hash_map_cache:
                            print_mp(
                                f"Skipping join_use_perfect_hash: {join_use_perfect_hash}, join_use_hash_map_cache: {join_use_hash_map_cache} because hash map caching should always be used except with perfect hashing",
                                is_root_rank,
                            )
                            continue

                        # Perfect hash join is disabled for substrait plans, see: https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/161
                        if query_source == "substrait" and (
                            join_use_perfect_hash or aggregation_use_perfect_hash
                        ):
                            print_mp(
                                f"Skipping join_use_perfect_hash: {join_use_perfect_hash}, aggregation_use_perfect_hash: {aggregation_use_perfect_hash}, query_source: {query_source}. Because, perfect hash is only manually enabled in physical plans",
                                is_root_rank,
                            )
                            continue

                        parameters.append(
                            QueryExecutionContext(
                                num_workers,
                                num_partitions,
                                use_overlap_mtx,
                                join_use_hash_map_cache,
                                read_use_zero_copy,
                                join_use_unique_keys,
                                join_use_perfect_hash,
                                join_use_mark_join,
                                use_partition_pruning,
                                filter_use_like_shift_and,
                                aggregation_use_perfect_hash,
                                query_info_ctx=query_info_ctx,
                            )
                        )
            # group same-query sets together to minimize data reloads
            parameters.sort(key=lambda p: p.query_info_ctx.query_idx)
            # We use the non-sandboxing path if sandboxing is not set, or if multiprocessing/multigpu is set.
            if not args.sandboxing or args.multiprocess:
                print_mp("Running experiments without subprocess sandbox", is_root_rank)
                run_tpc(
                    cat_ctx,
                    data_info,
                    scale_factor,
                    parameters,
                    edb_file,
                    edb_info,
                    all_errors,
                    args.repeat,
                    is_root_rank,
                    args.multiprocess,
                    multiprocess_runtime_context,
                    args.verify_results,
                )

            # Subprocess Case
            # Ideal case: spawn process, run to completion, join.
            # If subprocess fails, we iterate and launch the process again w/ the remaining parameters.
            # Subprocess MUST remove tasks from list for this to function correctly.
            else:
                subproc_ctx = subprocessing.get_context("spawn")
                with subproc_ctx.Manager() as manager:
                    parameter_queue = manager.list(parameters)
                    # explicitly replace errors list with list proxy
                    errors = manager.list()
                    print_mp(
                        "Running experiments with subprocess sandbox", is_root_rank
                    )
                    num_starting_params = len(parameter_queue)
                    iters = 0
                    while parameter_queue:
                        parent_pipe, child_pipe = subproc_ctx.Pipe()
                        subproc = subproc_ctx.Process(
                            target=run_tpc,
                            args=(
                                cat_ctx,
                                data_info,
                                scale_factor,
                                parameter_queue,
                                edb_file,
                                edb_info,
                                errors,
                                args.repeat,
                                is_root_rank,
                                args.multiprocess,
                                multiprocess_runtime_context,
                                args.verify_results,
                                child_pipe,
                            ),
                        )
                        subprocess_run(
                            subproc,
                            parameter_queue,
                            load_all_data,
                            is_root_rank,
                            parent_pipe,
                            args,
                        )
                    # convert to regular list before manager destructs
                    all_errors += list(errors)
        # Return should match outer loop indentation
        return all_errors

    errors = []
    errors = run_sweep(edb_file, edb_info, args)
    print_mp(f"Finished SQLite file at {edb_file}", is_root_rank)

    if errors:
        print_mp(
            "The following configurations run successfully but produce incorrect results",
            is_root_rank,
        )
        print_mp(errors, is_root_rank)

    if args.multiprocess:
        multiprocess_runtime_context.finalize()
        if args.storage_kind == "boost_shared_memory" and lib.mpi_rank() == 0:
            lib.finalize_shared_memory()
        lib.mpi_finalize()

    if errors:
        sys.exit(1)


def subprocess_kill(subproc, message, is_root_rank):
    print_mp(message, is_root_rank)
    subproc.terminate()
    time.sleep(1)
    # If process is still alive, try to kill more forcefully.
    if subproc.is_alive():
        print_mp("Process still alive after terminate, trying sigkill", is_root_rank)
        subproc.kill()
        time.sleep(1)
        # if subprocess is still alive after here something is really bad
        # TODO investigate ways to mitigate if it becomes an issue


def do_poll(subproc, pipe, timeout):
    start = time.monotonic()
    current = time.monotonic()
    poll_time = 1
    result = False
    # Iterate while timeout isn't reached, experiments are still in queue, and the subprocess is alive.
    while current - start < timeout and subproc.is_alive() and not result:
        result = pipe.poll(poll_time)
        current = time.monotonic()
    elapsed = current - start
    return result, elapsed


def subprocess_run(subproc, parameter_queue, load_all_data, is_root_rank, pipe, args):
    query_timeout = args.query_timeout
    data_timeout = args.data_timeout
    prev_items = len(parameter_queue)
    print(
        f"Starting parameter set execution with {len(parameter_queue)} sets remaining"
    )
    print(f"Using {query_timeout}s query timeout and {data_timeout}s data load timeout")
    subproc.start()

    killed = False
    iters = 0
    # Subprocess must pop items from queue for this to work properly
    while parameter_queue and not killed:
        print_mp(
            f"Parameter sets remaining: {len(parameter_queue)}",
            is_root_rank,
        )
        iters += 1
        # Stage 1: Wait on data load. Two Cases
        # load_all_data = True; the first iter, this determines success. After, we always expect success (no data to load).
        # load_all_data = False; we don't always load data, so we get true if it succeeded or didn't load, otherwise false.
        data_avail, elapsed = do_poll(subproc, pipe, data_timeout)
        print_mp(f"Data load stage ended after {elapsed:.2f}s", is_root_rank)
        if not data_avail:
            if elapsed >= data_timeout:
                subprocess_kill(
                    subproc,
                    f"Timeout triggered - data load did not complete within {data_timeout} seconds",
                    is_root_rank,
                )
                killed = True
                if load_all_data:
                    print_mp(
                        "Because this is load_all_data, dropping dependent experiments",
                        is_root_rank,
                    )
                    parameters[:] = []
            # the other option is we're not alive, so we break anyway
            break
        # if we get a false on receive, it means data loading failed for some reason
        if not pipe.recv():
            # Since load_all_data failed, we need to discard the experiments or we will not make forward progress.
            if load_all_data:
                parameters[:] = []
                break
            # if loading only query data, safe to proceed to next query and try again
            else:
                continue

        # Stage 2: wait on context build
        data_avail, elapsed = do_poll(subproc, pipe, query_timeout)
        print_mp(f"Context creation ended after {elapsed:.2f}s", is_root_rank)
        if not data_avail:
            if elapsed >= query_timeout:
                killed = True
                subprocess_kill(
                    subproc,
                    f"Timeout triggered - query did not complete within {query_timeout} seconds",
                    is_root_rank,
                )
            break
        # If pipe sends false, it generally means a query error or query validation failed.
        if not pipe.recv():
            # Both multiprocessing and single gpu break out of query or query validation if error.
            break

        # Stage 3: Wait on query completion
        for i in range(args.repeat):
            # Wait on query process
            data_avail, elapsed = do_poll(subproc, pipe, query_timeout)
            print_mp(
                f"Query execution/validation iteration {i} ended after {elapsed:.2f}s",
                is_root_rank,
            )
            if not data_avail:
                if elapsed >= query_timeout:
                    killed = True
                    subprocess_kill(
                        subproc,
                        f"Timeout triggered - query did not complete within {query_timeout} seconds",
                        is_root_rank,
                    )
                break
            # If pipe sends false, it generally means a query error or query validation failed.
            if not pipe.recv():
                # Both multiprocessing and single gpu break out of query or query validation if error.
                break
    subproc.join()

    # Breaking above will always take you here, which evaluates the subprocess execution status.
    if subproc.exitcode != 0:
        print_mp(
            f"Subprocess exited with non-zero return code with {len(parameter_queue)} tasks remaining.",
            is_root_rank,
        )
        print_mp(f"Subprocess Exit Code {subproc.exitcode}", is_root_rank)
        # We won't print the "may have" part if we deliberately killed the subprocess ourselves.
        if subproc.exitcode < 0 and not killed:
            print_mp(
                f"Subprocess may have been killed by {signal.Signals(-subproc.exitcode).name}",
                is_root_rank,
            )


if __name__ == "__main__":
    main()
