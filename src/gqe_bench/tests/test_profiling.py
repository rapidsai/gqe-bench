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
Tests for Profiler and the nsys launch/start/stop commands.

Usage:
    pytest gqe_bench/tests/test_profiling.py -v
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gqe_bench.profiling import NsysError, NsysProfiler


@pytest.fixture
def profiler(tmp_path: Path) -> NsysProfiler:
    """Profiler writing to tmp_path; "sf1k" is the prefix in expected names."""
    return NsysProfiler(
        binary=Path("/usr/bin/nsys"),
        output_dir=tmp_path,
        prefix="sf1k",
        trace="cuda,nvtx",
        launch_args="",
        extra_args="",
    )


class TestConstruction:
    def test_creates_missing_output_dir(self, tmp_path: Path) -> None:
        """nsys does not create it and fails at launch without it."""
        target = tmp_path / "does" / "not" / "exist"
        NsysProfiler(Path("/usr/bin/nsys"), target, "sf1k", "cuda,nvtx", "", "")
        assert target.is_dir()


class TestLaunchPrefix:
    def test_names_a_new_session(self, profiler: NsysProfiler) -> None:
        """--session-new opens the session the start/stop pair later drives."""
        argv = profiler.launch_prefix()
        assert argv[1] == "launch"
        assert f"--session-new={profiler.session}" in argv

    def test_launch_args_go_to_the_launch_verb(self, tmp_path: Path) -> None:
        """nsys takes --cuda-memory-usage on `launch` and rejects it on `start`."""
        profiler = NsysProfiler(
            Path("/usr/bin/nsys"), tmp_path, "sf1k", "cuda,nvtx", "--cuda-memory-usage=true", ""
        )
        assert "--cuda-memory-usage=true" in profiler.launch_prefix()


class TestVerifyArgs:
    """nsys takes both flag families on `profile`, so one run over `true` checks them."""

    def test_profiles_a_trivial_application(self) -> None:
        with patch("gqe_bench.profiling.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stderr="")
            NsysProfiler.verify_args(
                Path("/usr/bin/nsys"), "cuda,nvtx", "--cuda-memory-usage=true", "--sample=cpu"
            )
        argv = run.call_args[0][0]
        assert argv[1] == "profile"
        assert "--cuda-memory-usage=true" in argv
        assert "--sample=cpu" in argv
        assert argv[-1] == "true"

    def test_rejection_raises_with_stderr(self) -> None:
        with patch("gqe_bench.profiling.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stderr="unrecognised option '--bogus'")
            with pytest.raises(NsysError, match="unrecognised option"):
                NsysProfiler.verify_args(Path("/usr/bin/nsys"), "cuda,nvtx", "", "--bogus")


class TestStartStop:
    @staticmethod
    def _start(profiler: NsysProfiler, label: str = "6_e12") -> MagicMock:
        with patch("gqe_bench.profiling.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stderr="")
            profiler.start(label)
        return run

    def test_report_named_before_it_exists(self, profiler: NsysProfiler, tmp_path: Path) -> None:
        argv = self._start(profiler).call_args[0][0]
        assert argv[1] == "start"
        assert argv[argv.index("-o") + 1] == str(tmp_path / "sf1k_6_e12_%n")

    def test_stop_closes_the_session(self, profiler: NsysProfiler) -> None:
        self._start(profiler)
        with patch("gqe_bench.profiling.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stderr="")
            profiler.stop()
        assert run.call_args[0][0][1] == "stop"

    def test_start_failure_raises_with_stderr(self, profiler: NsysProfiler) -> None:
        """Collecting nothing must not pass silently: it wastes the whole sweep."""
        with patch("gqe_bench.profiling.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stderr="no such session")
            with pytest.raises(NsysError, match="no such session"):
                profiler.start("6_e12")

    def test_stop_failure_is_logged_not_raised(
        self, profiler: NsysProfiler, caplog: pytest.LogCaptureFixture
    ) -> None:
        """stop() runs from a finally; raising would discard the real exception."""
        self._start(profiler)
        with patch("gqe_bench.profiling.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stderr="session died")
            with caplog.at_level(logging.WARNING, logger="gqe_bench.profiling"):
                profiler.stop()  # must not raise
        assert "session died" in caplog.text

    def test_no_prefix_gives_the_bare_name(self, tmp_path: Path) -> None:
        """--nsys-prefix is optional; without it the label alone names the report."""
        unprefixed = NsysProfiler(Path("/usr/bin/nsys"), tmp_path, "", "cuda,nvtx", "", "")
        argv = self._start(unprefixed, "6_e12").call_args[0][0]
        assert argv[argv.index("-o") + 1] == str(tmp_path / "6_e12_%n")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
