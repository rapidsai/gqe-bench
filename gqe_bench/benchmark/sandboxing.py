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
import inspect
import os
import pathlib
import pickle
import selectors
import signal
import tempfile
import time
from subprocess import Popen
from typing import BinaryIO

from gqe_bench.benchmark import run
from gqe_bench.benchmark.run import (
    QueryError,
    clear_query_from_queue,
    clear_queue,
    is_unrecoverable_error,
    print_mp,
)


def run_sandboxed(
    run_suite_args: inspect.BoundArguments,
    parameters: list,
    load_all_data: bool,
    scale_factor: float,
    errors: list,
    invalid_results: list,
    args: argparse.Namespace,
):
    if args.num_ranks == 1:
        print_mp("Running experiments using single-gpu with sandboxing", not args.quiet)
    elif not args.sandboxing:
        print_mp(
            "Running experiments using MPI multi-gpu without sandboxing",
            not args.quiet,
        )
    else:
        print_mp(
            "Running experiments using MPI multi-gpu with sandboxing",
            not args.quiet,
        )
    # track if process had to quit due to unrecoverable error
    unrecoverable = False
    while parameters and not unrecoverable:
        tmpdir = tempfile.mkdtemp()
        # create a pipe to exchange with root mpi process
        pipe_path = os.path.join(tmpdir, "gqe_sandbox_pipe")
        os.mkfifo(pipe_path)
        # this pipe is only read by this process; root worker sends us heartbeat
        with open(pipe_path, "r+b", buffering=0) as pipe:
            # presently we also build worker pipes for each rank, but we only use them to distribute parameters initially for the time being
            worker_pipes = []
            for i in range(args.num_ranks):
                worker_pipe_path = f"{pipe_path}{i}"
                os.mkfifo(worker_pipe_path)
                worker_pipes.append(open(worker_pipe_path, "w+b", buffering=0))
                print(f"main creating worker pipe path named {worker_pipe_path}")
            process = launch_processes(pipe_path, load_all_data, scale_factor, args)

            print(f"main opening root pipe path named {pipe_path}")
            for worker_pipe in worker_pipes:
                pickle.dump(run_suite_args.arguments, worker_pipe)
            print("main continuing to execution monitoring")
            monitor_sandbox(process, parameters, pipe, load_all_data, errors, invalid_results, args)
            # we need to do the same in both cases, just give up after one try
            os.remove(pipe_path)
            for i, wp in enumerate(worker_pipes):
                wp.close()
                os.remove(f"{pipe_path}{i}")
            os.rmdir(tmpdir)
            if not args.sandboxing:
                break


def sandbox_kill(subproc: Popen, message: str):
    gid = subproc.pid
    print(message)
    subproc.terminate()
    start = time.monotonic()
    current = time.monotonic()
    timeout = 60
    print(f"waiting up to {timeout}s for process to die")
    # similar to subprocess_kill but the api is different...
    while current - start < timeout and subproc.poll() is None:
        time.sleep(1)
        current = time.monotonic()
    if subproc.poll() is None:
        print("Process still alive after terminate, trying sigkill")
        subproc.kill()
        os.killpg(gid, 9)
    # above we killed mpirun and its children, here we send sig 0 to check if the process group still exists
    # we do this because memory takes forever to deallocate with mpi and we can end up bumping into our own dead processes trying to restart
    # if we got here, this should succeed, but we set a limit just in case.
    for i in range(timeout):
        try:
            os.killpg(gid, 0)
            time.sleep(1)
        # when error, process group no longer exists
        except OSError:
            break
    else:
        # If we didn't exit above, the behavior is undefined; we probably cannot recover from a process we cannot kill.
        print("Giving up on waiting for process group to die - system may be unrecoverable")
    # give a small buffer for OS cleanup before we start new processes
    time.sleep(3)


def sandbox_poll(subproc: Popen, pipe: BinaryIO, timeout: int):
    fd = pipe.fileno()
    start = time.monotonic()
    current = time.monotonic()
    poll_time = 1
    result = False
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        # Iterate while timeout isn't reached, experiments are still in queue, and the subprocess is alive.
        while current - start < timeout and subproc.poll() is None and not result:
            # poll
            result = selector.select(timeout=poll_time)
            current = time.monotonic()
        elapsed = current - start
    return result, elapsed


def launch_processes(
    pipe_path: str, load_all_data: bool, scale_factor: float, args: argparse.Namespace
):
    # execute run.py as main of new python process
    run_path = pathlib.Path(run.__file__).resolve()
    is_mp = args.num_ranks > 1
    cmd = [
        "python3",
        "-u",
        f"{run_path}",
        f"{pipe_path}",
        f"{load_all_data}",
        f"{scale_factor}",
        f"{args.storage_kind}",
        f"{args.boost_pool_size}",
        f"{is_mp}",
    ]
    if is_mp:
        # prepend MPI to standard command
        cmd = [
            "mpirun",
            "--allow-run-as-root",
            "--tag-output",
            "-n",
            f"{args.num_ranks}",
        ] + cmd
    print_mp(f"Running processes as: {' '.join(cmd)}", not args.quiet)
    process = Popen(
        cmd,
        env=os.environ.copy(),
        cwd=os.getcwd(),  # stdout=subprocess.PIPE, stderr=subprocess.PIPE
        start_new_session=True,
    )
    return process


# we expect a dictionary to be sent containing an error ID and info to log
def parse_sandbox_errors(data: dict, errors: list, invalid_results: list):
    unrecoverable = False
    for key, value in data.items():
        if key == QueryError.validation:
            invalid_results.append(value)
        else:
            errors.append(value)
        if is_unrecoverable_error(value):
            unrecoverable = True
    return unrecoverable


def monitor_sandbox(
    process: Popen,
    parameter_queue: list,
    pipe: BinaryIO,
    load_all_data: bool,
    errors: list,
    invalid_results: list,
    args: argparse.Namespace,
):
    query_timeout = args.query_timeout
    data_timeout = args.data_timeout
    print_mp(
        f"Starting parameter set execution with {len(parameter_queue)} sets remaining",
        not args.quiet,
    )
    print_mp(
        f"Using {query_timeout}s query timeout and {data_timeout}s data load timeout",
        not args.quiet,
    )

    unrecoverable = False
    while parameter_queue and not unrecoverable:
        print_mp(f"Parameter sets remaining: {len(parameter_queue)}", not args.quiet)

        # first thing we do is pop on other side; if we fail, we dump params, so no big deal to pop presumptively
        parameter = parameter_queue.pop(0)
        # Stage 1: Wait on data load. Two Cases
        # load_all_data = True; the first iter, this determines success. After, we always expect success (no data to load).
        # load_all_data = False; we don't always load data, so we get true if it succeeded or didn't load, otherwise false.
        data_avail, elapsed = sandbox_poll(process, pipe, data_timeout)
        print_mp(f"Data load stage ended after {elapsed:.2f}s", not args.quiet)
        if not data_avail:
            if elapsed >= data_timeout:
                kill_msg = (
                    f"Timeout triggered - data load did not complete within {data_timeout} seconds"
                )
            else:
                kill_msg = f"No data available but timeout not triggered; likely fatal error; attempting kill and reset"

            sandbox_kill(process, kill_msg)
            killed = True
            if load_all_data:
                clear_queue(parameter_queue)
            else:
                clear_query_from_queue(parameter, parameter_queue)
                # we killed here so always break and get a new process
            break
        # if we get a false on receive, it means data loading failed for some reason
        if not pickle.load(pipe):
            unrecoverable = parse_sandbox_errors(pickle.load(pipe), errors, invalid_results)
            # Since load_all_data failed, we need to discard the experiments or we will not make forward progress.
            if load_all_data:
                clear_queue(parameter_queue)
                break
            else:
                clear_query_from_queue(parameter, parameter_queue)
                if unrecoverable:
                    break
                # if loading only query data, safe to proceed to next query and try again
                else:
                    continue

        # Stage 2: wait on context build
        data_avail, elapsed = sandbox_poll(process, pipe, query_timeout)
        print_mp(f"Context creation ended after {elapsed:.2f}s", not args.quiet)
        if not data_avail:
            killed = True
            if elapsed >= query_timeout:
                kill_msg = (
                    f"Timeout triggered - query did not complete within {query_timeout} seconds"
                )
            else:
                kill_msg = f"No data available but timeout not triggered; likely fatal error; attempting kill and reset"
            sandbox_kill(process, kill_msg)
            # always break if we get here to move on to new process
            break
        # If pipe sends false, it means we failed to load context and move on to next parameter set.
        if not pickle.load(pipe):
            unrecoverable = parse_sandbox_errors(pickle.load(pipe), errors, invalid_results)
            if unrecoverable:
                break
            continue

        # Stage 3: Wait on query completion
        for i in range(args.repeat):
            # Wait on query process
            data_avail, elapsed = sandbox_poll(process, pipe, query_timeout)
            print_mp(
                f"Query execution/validation iteration {i} ended after {elapsed:.2f}s",
                not args.quiet,
            )
            if not data_avail:
                killed = True
                if elapsed >= query_timeout:
                    kill_msg = (
                        f"Timeout triggered - query did not complete within {query_timeout} seconds"
                    )
                else:
                    kill_msg = f"No data available but timeout not triggered; likely fatal error; attempting kill and reset"
                sandbox_kill(process, kill_msg)
                # always break if we get here to move on to new process
                break
            # If pipe sends false, it generally means a query error or query validation failed.
            if not pickle.load(pipe):
                unrecoverable = parse_sandbox_errors(pickle.load(pipe), errors, invalid_results)
                # Both multiprocessing and single gpu break out of query or query validation if error.
                # Note that this only breaks the inner loop - all other cases continue unless unrecoverable
                break
    exit_code = process.wait()

    # Breaking above will always take you here, which evaluates the process execution status.
    if exit_code != 0:
        print_mp(
            f"Subprocess exited with non-zero return code with {len(parameter_queue)} tasks remaining.",
            not args.quiet,
        )
        print_mp(f"Subprocess Exit Code {exit_code}", not args.quiet)
        # We won't print the "may have" part if we deliberately killed the process ourselves.
        if exit_code < 0 and not killed:
            print_mp(
                f"Subprocess may have been killed by {signal.Signals(-exit_code).name}",
                not args.quiet,
            )
