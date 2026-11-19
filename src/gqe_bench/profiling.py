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

"""Profiling of the engine, one report per collection window.

A ``Profiler`` offers a prefix for the engine's launch argv and a start/stop
pair around the work to profile. The caller sets the window, because only it
knows where the work begins and ends. ``NoopProfiler`` collects nothing, so
callers need no conditional for the profiling-off case.

``NsysProfiler`` runs the engine under ``nsys launch --session-new``, which
starts it with collection unarmed, then arms it with ``nsys start -o NAME`` and
closes it with ``nsys stop``. Because collection is armed only around the work
of interest, data loading never lands in a report, and ``-o`` names each report
before it exists, so no report has to be matched to its work afterwards.
"""

import logging
import os
import shlex
import subprocess
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class NsysError(RuntimeError):
    """Raised when an nsys control command fails."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"nsys exited with code {returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr: {stderr}"
        )


class Profiler(ABC):
    """Drives an out-of-process profiler around the engine."""

    @abstractmethod
    def launch_prefix(self) -> list[str]:
        """argv to prepend to the engine launch."""

    @abstractmethod
    def start(self, label: str) -> None:
        """Begin collecting into a report named after ``label``."""

    @abstractmethod
    def stop(self) -> None:
        """Finish the report opened by ``start``."""


class NoopProfiler(Profiler):
    """Collects nothing: the behaviour when no profiler was requested."""

    def launch_prefix(self) -> list[str]:
        return []

    def start(self, label: str) -> None:
        pass

    def stop(self) -> None:
        pass


class NsysProfiler(Profiler):
    """Collects one Nsight Systems report per start/stop window."""

    @classmethod
    def verify_args(cls, binary: Path, trace: str, launch_args: str, extra_args: str) -> None:
        """Reject arguments a sweep would fail on, before it starts.

        ``nsys profile`` takes the launch and the collection flags together and
        parses them before running the application, so profiling ``true``
        accepts exactly what a sweep would. The report is written into a
        temporary directory and discarded with it.
        """
        with tempfile.TemporaryDirectory() as scratch:
            cls._nsys(
                binary,
                [
                    "profile",
                    "--trace",
                    trace,
                    *shlex.split(launch_args),
                    *shlex.split(extra_args),
                    "-o",
                    str(Path(scratch) / "verify"),
                    "true",
                ],
            )

    def __init__(
        self,
        binary: Path,
        output_dir: Path,
        prefix: str,
        trace: str,
        launch_args: str,
        extra_args: str,
    ) -> None:
        """Bind an nsys binary to an output location.

        ``binary`` must already name an executable. ``trace`` names the APIs to
        record and belongs to the launch, as does ``launch_args`` (e.g.
        ``--cuda-memory-usage``); ``extra_args`` is a single string of
        collection flags, split like a shell would, and belongs to the
        collection. nsys accepts each only on its own verb.
        """
        self.binary = binary
        self.output_dir = output_dir
        # nsys does not create the -o directory and fails at launch without it.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.trace = trace
        self.launch_args = tuple(shlex.split(launch_args))
        self.extra_args = tuple(shlex.split(extra_args))
        # Report opened by start(), so stop() can name it if nsys refuses.
        self._target: Path | None = None
        # One session per engine launch. A session dies with the process it was
        # created for, so a server restart needs a new one.
        self.session = f"gqe_bench_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    def launch_prefix(self) -> list[str]:
        """argv that runs the engine under nsys with collection unarmed."""
        return [
            str(self.binary),
            "launch",
            f"--session-new={self.session}",
            "--trace",
            self.trace,
            *self.launch_args,
        ]

    def start(self, label: str) -> None:
        """Arm collection into ``<prefix>_<label>_%n``, or ``<label>_%n`` with no prefix.

        ``%n`` is nsys's own counter, the lowest integer the directory does not
        already hold, so a re-run adds reports beside the earlier ones instead
        of overwriting them. nsys appends its own ``.nsys-rep``.
        """
        base = f"{self.prefix}_{label}" if self.prefix else label
        self._target = self.output_dir / f"{base}_%n"
        self._nsys(
            self.binary,
            [
                "start",
                f"--session={self.session}",
                *self.extra_args,
                "-o",
                str(self._target),
            ],
        )

    def stop(self) -> None:
        """Close the report, logging rather than raising if nsys refuses.

        The caller stops from a ``finally``, where raising would discard the
        exception that got it there. One way nsys refuses: the engine crashed,
        taking its session with it.
        """
        name = self._target.name if self._target else "<unnamed>"
        self._target = None
        try:
            self._nsys(self.binary, ["stop", f"--session={self.session}"])
        except NsysError as e:
            logger.warning("Profile %s not written: %s", name, e)

    @staticmethod
    def _nsys(binary: Path, args: list[str]) -> None:
        """Run an nsys command, raising NsysError if it fails."""
        cmd = [str(binary), *args]
        logger.debug("nsys cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise NsysError(cmd, result.returncode, result.stderr.strip())
