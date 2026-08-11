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

"""GqeServer wrapper — manages the gqe_node_manager subprocess lifetime."""

import logging
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

_SHM_PATH = Path("/dev/shm/gqe_shared_memory")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50051
DEFAULT_ADDR = f"{DEFAULT_HOST}:{DEFAULT_PORT}"
DEFAULT_SERVER_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

# Wall-clock timeout for gqe_node_manager to begin accepting connections.
# Sized to exceed every gqe-side bootstrap deadline so the C++ side's own
# error reporting fires first on a genuine bootstrap failure.
_START_TIMEOUT_S = 300

# gRPC deadline the node manager applies to each task manager's ExecutePlan
# RPC. Sized for long-running sweep queries, which exceed the engine's own
# default and surface as "Task manager rank N failed: Deadline Exceeded".
_QUERY_TIMEOUT_S = 7200


class ServerAlreadyRunning(RuntimeError):
    """A GqeServer instance already holds the singleton slot."""


class ServerStartFailed(RuntimeError):
    """gqe_node_manager did not become ready (exited early or timed out)."""


class GqeServer:
    """Context manager that manages a gqe_node_manager process.

    Use as a context manager only. Only one instance may exist at a time;
    the slot is released when the context exits OR when any construction /
    startup path raises.
    """

    _instance: ClassVar["GqeServer | None"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "GqeServer":
        if cls._instance is not None:
            raise ServerAlreadyRunning("Only one GqeServer may exist at a time")
        return super().__new__(cls)

    def __init__(
        self,
        server_bin: Path | None,
        task_manager_bin: Path | None,
        addr: str = DEFAULT_ADDR,
        num_gpus: int = 1,
        env: dict[str, str] | None = None,
    ) -> None:
        """Validate binary paths and claim the singleton slot.

        Raises ValueError if either binary path is None.
        """
        if server_bin is None:
            raise ValueError("server binary path not provided")
        if task_manager_bin is None:
            raise ValueError("task manager binary path not provided")
        self._bin = server_bin
        self._task_manager_bin = task_manager_bin
        self._addr = addr
        self._num_gpus = num_gpus
        self._env = env
        self._process: subprocess.Popen | None = None
        # Claim the singleton slot only after all validation has succeeded.
        # If __init__ raises before this line, the slot stays free for the
        # next GqeServer(...) attempt.
        GqeServer._instance = self

    @staticmethod
    def _cleanup_shared_memory() -> None:
        """Remove stale shared memory segment left by a previous node_manager."""
        try:
            _SHM_PATH.unlink()
        except FileNotFoundError:
            pass

    def _start(self) -> None:
        """Launch the server and wait until the port is accepting connections.

        Bounded by _START_TIMEOUT_S. Raises ServerStartFailed on either
        subprocess-exited-early or timeout-without-listener. On timeout the
        subprocess is terminated before the exception is raised.
        """
        self._cleanup_shared_memory()
        host, port_str = self._addr.rsplit(":", 1)
        server_argv = [
            str(self._bin),
            "--address",
            host,
            "--port",
            port_str,
            "--num-gpus",
            str(self._num_gpus),
            "--task-manager-binary",
            str(self._task_manager_bin),
            "--query-timeout",
            str(_QUERY_TIMEOUT_S),
        ]
        logger.info("Starting server on %s", self._addr)
        logger.debug("server argv: %s", " ".join(server_argv))
        self._process = subprocess.Popen(
            server_argv,
            env=self._env,
        )
        port = int(port_str)
        deadline = time.monotonic() + _START_TIMEOUT_S

        while True:
            if self._process.poll() is not None:
                raise ServerStartFailed(
                    f"gqe_node_manager exited with code {self._process.returncode} "
                    f"before becoming ready"
                )
            if time.monotonic() > deadline:
                # SIGTERM not SIGKILL: lets node_manager's pre-Arrow signal
                # handler reap its task_manager subtree. SIGKILL bypasses it.
                self._process.send_signal(signal.SIGTERM)
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=10)
                raise ServerStartFailed(
                    f"gqe_node_manager did not become ready within {_START_TIMEOUT_S}s"
                )
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    logger.info("Server ready on %s", self._addr)
                    return
            except OSError:
                time.sleep(0.1)

    def _stop(self) -> None:
        """Terminate the server process and clean up shared memory."""
        try:
            if self._process is None or self._process.poll() is not None:
                return
            logger.info("Stopping server (SIGTERM) on %s", self._addr)
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=60)
        finally:
            self._cleanup_shared_memory()

    def is_alive(self) -> bool:
        """True if the server process is still running."""
        return self._process is not None and self._process.poll() is None

    @property
    def addr(self) -> str:
        return self._addr

    def __enter__(self) -> "GqeServer":
        """Start the subprocess; block until it accepts connections.

        Python does not call __exit__ if __enter__ raises, so we must clean
        up manually. The complication: _stop() can itself raise
        (ProcessLookupError, TimeoutExpired from a SIGKILL that doesn't
        take). A bare `raise` after `_stop()` would lose the original
        _start() exception if _stop() throws first. `raise start_exc from
        stop_exc` preserves both: the caller sees the original failure as
        the primary exception, with the cleanup failure chained as __cause__.

        The singleton slot is released in the outer `finally` so it resets
        on every failure path — start-only-failed, start-and-stop-both-failed
        — and a subsequent GqeServer(...) attempt is not blocked by the
        prior attempt's leaked slot.
        """
        try:
            self._start()
        except BaseException as start_exc:
            try:
                try:
                    self._stop()
                except Exception as stop_exc:
                    raise start_exc from stop_exc
                raise start_exc
            finally:
                GqeServer._instance = None
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Stop the subprocess and release the singleton slot.

        The slot is released in `finally` even if `_stop()` raises; otherwise
        ServerAlreadyRunning would block any caller-side restart.
        """
        try:
            self._stop()
        finally:
            GqeServer._instance = None
