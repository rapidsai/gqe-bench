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

"""Record rendering for gqe_bench.

Records go to stderr in the line format the node manager's log drain applies to
every engine record:

    [YYYY-MM-DD HH:MM:SS.ffffff] [<source>] [<level>] [thread <tid>] <message>

The engine inherits this process's descriptors, so its records and these share
one stream and interleave in emission order. The `gqe-bench:<module>` source
distinguishes them from the engine's `server` and `gpu<rank>`.

The plugin renders the same format from C++; see nvtx_plugin/log.hpp.
"""

import logging
import sys
import threading
import time
from argparse import Namespace

# Valid --log-level values mapped to logging thresholds. QUIET maps to None,
# the sentinel for the logging.disable() off switch rather than a threshold.
LOG_LEVELS: dict[str, int | None] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "QUIET": None,
}


# Python level names mapped to the names the drain writes. The drain names its
# own levels rather than deferring to spdlog, so `warning` has no counterpart
# there; the rest coincide.
_LEVEL_NAMES: dict[str, str] = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warn",
    "ERROR": "error",
    "CRITICAL": "critical",
}


class _SpdlogFormatter(logging.Formatter):
    """Render records in the line format the node manager's log drain applies.

    The attributes the format string reads are restored after formatting because
    a LogRecord is shared by every handler that receives it.
    """

    _FMT = (
        "[%(asctime)s] [gqe-bench:%(module)s] [%(levelname)s] "
        "[thread %(native_thread)d] %(message)s"
    )
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Render the timestamp to the microsecond precision the drain emits.

        `datefmt` reaches `time.strftime`, which has no sub-second directive, so
        the fractional part is appended rather than carried in the format.
        """
        seconds = time.strftime(datefmt or self._DATEFMT, self.converter(record.created))
        return f"{seconds}.{int(record.created % 1 * 1_000_000):06d}"

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        record.levelname = _LEVEL_NAMES[original]
        # The drain reports the kernel thread id, which `record.thread` is not.
        # Handlers format on the thread that emitted the record.
        record.native_thread = threading.get_native_id()
        try:
            return super().format(record)
        finally:
            record.levelname = original
            del record.native_thread


def configure_logging(args: Namespace) -> None:
    """Apply the parsed log_level: a basicConfig threshold, or full off for QUIET.

    QUIET uses logging.disable(logging.CRITICAL), the global off switch, rather
    than a threshold. Every other level sets the root logger threshold via
    basicConfig. Expects log_level already normalized by _validate_log_level.
    """
    level = LOG_LEVELS[args.log_level]
    if level is None:
        logging.disable(logging.CRITICAL)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_SpdlogFormatter())
    logging.basicConfig(level=level, handlers=[handler])
