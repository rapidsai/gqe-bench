#!/usr/bin/env python3
#
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

"""
Tests for the GqeServer wrapper.

Usage:
    pytest gqe_bench/tests/test_server.py -v
"""

import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gqe_bench.server import (
    DEFAULT_ADDR,
    GqeServer,
    ServerAlreadyRunning,
    ServerStartFailed,
)


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Release the singleton slot before and after each test."""
    GqeServer._instance = None
    yield
    GqeServer._instance = None


class TestGqeServer:
    """Unit tests for GqeServer."""

    def test_instantiation_and_addr(self) -> None:
        srv = GqeServer(Path("/bin/server"), Path("/bin/task_manager"))
        assert srv.addr == DEFAULT_ADDR

    def test_singleton_enforcement(self) -> None:
        GqeServer(Path("/bin/server"), Path("/bin/task_manager"))
        with pytest.raises(ServerAlreadyRunning, match="Only one GqeServer"):
            GqeServer(Path("/bin/server2"), Path("/bin/task_manager2"))

    def test_singleton_reject_is_runtime_error_subclass(self) -> None:
        """Back-compat: code catching bare RuntimeError still matches."""
        GqeServer(Path("/bin/server"), Path("/bin/task_manager"))
        with pytest.raises(RuntimeError):
            GqeServer(Path("/bin/server2"), Path("/bin/task_manager2"))

    def test_init_value_error_does_not_pin_singleton(self) -> None:
        """__init__ raising ValueError must not leak the singleton slot.

        Pre-existing bug: __new__ assigned _instance before __init__ ran,
        so a ValueError from __init__ (missing bin arg) pinned the singleton
        to a partially-initialized object. Fix moves the assignment to the
        end of __init__, after all validation succeeds.
        """
        with pytest.raises(ValueError):
            GqeServer(None, Path("/bin/task_manager"))
        assert GqeServer._instance is None
        # Subsequent valid construction must succeed.
        srv = GqeServer(Path("/bin/server"), Path("/bin/task_manager"))
        assert GqeServer._instance is srv


class TestGqeServerEdgeCases:
    """Tests for _stop() edge cases."""

    def test_stop_when_process_already_dead(self) -> None:
        srv = GqeServer(Path("/bin/server"), Path("/bin/task_manager"))
        proc = MagicMock()
        proc.poll.return_value = 0
        srv._process = proc
        srv._stop()
        proc.send_signal.assert_not_called()

    def test_exit_releases_singleton_even_if_stop_raises(self) -> None:
        """__exit__ must reset _instance even when _stop() raises.

        Regression: before this fix, a TimeoutExpired from the post-SIGKILL
        wait would propagate past _stop() and skip the _instance reset,
        pinning the singleton and causing every subsequent GqeServer(...)
        call to raise 'Only one GqeServer'.
        """
        import subprocess

        srv = GqeServer(Path("/bin/server"), Path("/bin/task_manager"))
        srv._stop = MagicMock(side_effect=subprocess.TimeoutExpired(cmd="x", timeout=60))
        with pytest.raises(subprocess.TimeoutExpired):
            srv.__exit__(None, None, None)
        assert GqeServer._instance is None

    def test_enter_releases_singleton_on_start_failure(self) -> None:
        """__enter__ must reset _instance when _start fails.

        Without the fix, a _start() failure on the first attempt would pin
        the singleton and every subsequent GqeSession.__enter__ (which does
        GqeServer(...)) would hit ServerAlreadyRunning for the rest of the
        process lifetime.
        """
        srv = GqeServer(Path("/bin/server"), Path("/bin/task_manager"))
        srv._start = MagicMock(side_effect=ServerStartFailed("nope"))
        srv._stop = MagicMock()
        with pytest.raises(ServerStartFailed):
            srv.__enter__()
        assert GqeServer._instance is None

    def test_enter_releases_singleton_when_both_start_and_stop_raise(self) -> None:
        """Pins the `finally` placement: the slot must reset on the chained-
        exception path (_start raises, then _stop also raises) not only the
        simple path (_start raises, _stop succeeds).
        """
        srv = GqeServer(Path("/bin/server"), Path("/bin/task_manager"))
        srv._start = MagicMock(side_effect=ServerStartFailed("start nope"))
        srv._stop = MagicMock(side_effect=RuntimeError("stop also nope"))
        with pytest.raises(ServerStartFailed):
            srv.__enter__()
        assert GqeServer._instance is None

    def test_start_times_out_and_terminates_subprocess(self) -> None:
        """_start must bound its readiness loop and SIGTERM a non-listening child.

        SIGTERM-not-SIGKILL matters: the C++ node_manager's pre-Arrow signal
        handler reaps its task_manager subtree on SIGTERM but is bypassed by
        SIGKILL.
        """

        srv = GqeServer(Path("/bin/server"), Path("/bin/task_manager"))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # alive throughout
        with (
            patch("gqe_bench.server.subprocess.Popen", return_value=mock_proc),
            patch(
                "gqe_bench.server.socket.create_connection",
                side_effect=OSError("no listener"),
            ),
            patch("gqe_bench.server._START_TIMEOUT_S", 0.1),
        ):
            with pytest.raises(ServerStartFailed, match="did not become ready"):
                srv._start()
        mock_proc.send_signal.assert_called_once_with(signal.SIGTERM)
        mock_proc.kill.assert_not_called()

    def test_start_timeout_falls_back_to_kill_if_terminate_hangs(self) -> None:
        """If SIGTERM does not drain the subprocess, _start escalates to SIGKILL."""
        import subprocess

        srv = GqeServer(Path("/bin/server"), Path("/bin/task_manager"))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # alive throughout
        # First wait (post-SIGTERM) hangs; second wait (post-SIGKILL) succeeds.
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="gqe_node_manager", timeout=10),
            None,
        ]
        with (
            patch("gqe_bench.server.subprocess.Popen", return_value=mock_proc),
            patch(
                "gqe_bench.server.socket.create_connection",
                side_effect=OSError("no listener"),
            ),
            patch("gqe_bench.server._START_TIMEOUT_S", 0.1),
        ):
            with pytest.raises(ServerStartFailed, match="did not become ready"):
                srv._start()
        mock_proc.send_signal.assert_called_once_with(signal.SIGTERM)
        mock_proc.kill.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
