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

"""JSON5 configuration and CLI argument parsing for gqe_bench.

All config fields are defined once in FIELDS. JSON validation, CLI argument
generation, and defaults are all driven from this single definition.

Sweep field defaults come from DDL DEFAULT clauses via PRAGMA table_info.
See system_under_test.sql for the authoritative default values.
"""

import copy
import logging
import shlex
import shutil
import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import json5

from gqe_bench._artifacts import CLI, NODE_MANAGER, TASK_MANAGER
from gqe_bench.logger import LOG_LEVELS
from gqe_bench.profiling import NsysError, NsysProfiler
from gqe_bench.query_source import QuerySource
from gqe_bench.schema import (
    DATA_INFO_MAPPING,
    EXECUTION_BACKENDS,
    QUERY_PARAMS_MAPPING,
    TableMapping,
)
from gqe_bench.server import DEFAULT_SERVER_URL
from gqe_bench.suites.tpch import TpchSuite

logger = logging.getLogger(__name__)


class FieldMode(StrEnum):
    """Field scope: SHARED (valid in any mode) or SWEEP (rejected in pretuned mode)."""

    SHARED = "shared"
    SWEEP = "sweep"


@dataclass(frozen=True)
class Field:
    """Single source of truth for a config/CLI parameter.

    Attributes:
        name:      Config/CLI key and Namespace attribute name.
        type:      Python type for argparse routing (str, int, bool, list, Path).
        default:   Default value. None means "must be provided or resolved."
                   Use ``get_default()`` to read.
        help:      Help text shown by --help and used for JSON config documentation.
        mode:      SHARED (valid in any mode) or SWEEP (rejected in pretuned mode).
        cli_flags: Argparse flags (e.g. ("--dataset",) or ("-p", "--num-partitions")).
        resolve:   Called post-parse when value is None. Signature: (Namespace) -> value.
        element_type: For list fields, argparse converter for each CLI value. None keeps
                   the values as strings.
        nargs:     For non-list, non-bool fields, argparse nargs. Set to "?" for a flag
                   whose value is optional.
        const:     For non-list, non-bool fields, value used when an nargs="?" flag is
                   given without one.
        requires:  Name of a field that must be set for this one to mean
                   anything. Providing this field without that one is an error.
                   A field with ``requires`` must default to None and get its real
                   default from ``resolve``.
    """

    name: str
    type: type
    default: Any
    help: str
    mode: FieldMode = FieldMode.SHARED
    cli_flags: tuple[str, ...] = ()
    resolve: Callable[[Namespace], Any] | None = None
    element_type: Callable[[str], Any] | None = None
    nargs: str | None = None
    const: Any = None
    requires: str | None = None

    def get_default(self) -> Any:
        """Return a deep copy of ``default``.

        Use this rather than reading ``.default`` directly: callers that place
        the result on a Namespace can safely mutate it without affecting other
        Namespaces or this Field's own default.
        """
        return copy.deepcopy(self.default)


# Resolve callbacks, called for fields still None after parsing. The binary
# resolvers return None if the binary is missing; the spawn site turns that
# into ArtifactMissing via Artifact.require() (gqe_bench/session.py).


def _resolve_server_bin(ns: Namespace) -> Path | None:
    """Resolve the gqe_node_manager binary from the installed package."""
    return NODE_MANAGER.locate()


def _resolve_task_manager_bin(ns: Namespace) -> Path | None:
    """Resolve the gqe_task_manager binary from the installed package."""
    return TASK_MANAGER.locate()


def _resolve_cli_bin(ns: Namespace) -> Path | None:
    """Resolve the gqe-cli binary from the installed package."""
    return CLI.locate()


def _resolve_nsys_output_dir(ns: Namespace) -> Path | None:
    """Default the profile directory to the working directory."""
    return Path.cwd() if ns.nsys else None


def _resolve_nsys_trace(ns: Namespace) -> str | None:
    """Default the traced APIs to CUDA and NVTX."""
    return "cuda,nvtx" if ns.nsys else None


def _resolve_nsys_launch_args(ns: Namespace) -> str | None:
    """Default to no extra launch flags."""
    return "" if ns.nsys else None


def _resolve_nsys_args(ns: Namespace) -> str | None:
    """Default to no extra collection flags."""
    return "" if ns.nsys else None


def _resolve_nsys_prefix(ns: Namespace) -> str | None:
    """Default to no filename prefix."""
    return "" if ns.nsys else None


def parse_bool(text: str) -> bool:
    """Parse a boolean CLI value.

    ``type=bool`` would not work, because ``bool("false")`` is True.
    """
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    raise ArgumentTypeError(f'Expected a Boolean value ("true" or "false"), got {text!r}')


def _cli_converter(mapping: TableMapping, name: str) -> Callable[[str], Any]:
    """Return the argparse converter for a sweep field, typed like its DDL column."""
    py_type = mapping.field_py_type(name)
    return parse_bool if py_type is bool else py_type


_HELP_OVERRIDES: dict[str, str] = {
    "shmem_policy": "Shared memory allocation policy used at server startup",
    "use_late_materialization": "Allow QEP late materialization; false forces early gathering. Ignored by legacy.",
    "join_materialization_threshold": (
        "QEP late materialization gathers a join output's columns from an input whose output "
        "rows divided by its rows are below this value, in [0, 1]; 0 disables. Ignored by legacy."
    ),
    "execution_backend": ("Backend used to execute queries: legacy_task_graph or qep_task_graph"),
    "use_partition_pruning": "Enable partition pruning (requires clustered dataset)",
    "compression_level": "Compression level (1-12). Only takes effect with use_cpu_compression.",
    "join_use_hash_map_cache": "Cache the join build-side hash map for multiple partitions",
}

_BOOL_PREFIXES = ("join_use_", "filter_use_", "aggregation_use_", "read_use_", "use_")


def _auto_help(name: str, is_bool: bool) -> str:
    """Generate help text for an auto-derived sweep field."""
    if name in _HELP_OVERRIDES:
        return _HELP_OVERRIDES[name]
    if is_bool:
        stripped = name
        for prefix in _BOOL_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                break
        return f"Enable {stripped.replace('_', ' ')}"
    return name.replace("_", " ").capitalize()


def _auto_fields(
    *mappings: TableMapping,
    explicit: frozenset[str],
) -> tuple[Field, ...]:
    """Generate Field entries for sweep params not in the explicit set.

    Defaults are read from DDL DEFAULT clauses via PRAGMA table_info.
    See system_under_test.sql for the authoritative default values.
    Fields without a DDL DEFAULT get default=None (must be provided by config).
    """
    fields: list[Field] = []
    for mapping in mappings:
        for name in mapping.sweep_fields:
            if name in explicit:
                continue
            raw_default = mapping.field_default(name)
            is_bool = mapping.is_bool_field(name)
            if raw_default is not None:
                sweep_default = [bool(raw_default)] if is_bool else [raw_default]
            else:
                sweep_default = None
            fields.append(
                Field(
                    name=name,
                    type=list,
                    default=sweep_default,
                    help=_auto_help(name, is_bool),
                    mode=FieldMode.SWEEP,
                    cli_flags=(f"--{name.replace('_', '-')}",),
                    element_type=_cli_converter(mapping, name),
                )
            )
    return tuple(fields)


_MANUAL_FIELDS: tuple[Field, ...] = (
    # Shared fields — valid in any mode
    Field(
        name="dataset",
        type=Path,
        default=None,
        help="Path to dataset directory",
        cli_flags=("--dataset",),
    ),
    Field(
        name="sql",
        type=Path,
        default=None,
        help="Path to query SQL directory",
        cli_flags=("--sql",),
    ),
    Field(
        name="solution",
        type=Path,
        default=None,
        help="Path to reference solutions directory (enables validation)",
        cli_flags=("--solution",),
    ),
    Field(
        name="schema",
        type=Path,
        default=None,
        help="Path to schema.sql (default: suite-specific built-in DDL, if available)",
        cli_flags=("--schema",),
    ),
    Field(
        name="queries",
        type=list,
        default=None,
        help="Query filter list",
        cli_flags=("-q", "--queries"),
    ),
    Field(
        name="output",
        type=Path,
        default=None,
        help="Output .db3 path",
        cli_flags=("-o", "--output"),
    ),
    Field(
        name="repeat",
        type=int,
        default=6,  # 1 warmup (r_number=0) + 5 timed
        help="Repetitions per config",
        cli_flags=("-rep", "--repeat"),
    ),
    Field(
        name="validate_dir",
        type=Path,
        default=None,
        help="Directory for validation temp files",
        cli_flags=("--validate-dir",),
    ),
    Field(
        name="suite_name",
        type=str,
        default=TpchSuite.NAME,
        help="Benchmark suite name",
        cli_flags=("--suite-name",),
    ),
    Field(
        name="server_url",
        type=str,
        default=DEFAULT_SERVER_URL,
        help="gqe-server URL",
        cli_flags=("--server-url",),
    ),
    Field(
        name="server_bin",
        type=Path,
        default=None,
        help="Path to gqe_node_manager binary",
        cli_flags=("--server-bin",),
        resolve=_resolve_server_bin,
    ),
    Field(
        name="cli_bin",
        type=Path,
        default=None,
        help="Path to gqe-cli binary",
        cli_flags=("--cli-bin",),
        resolve=_resolve_cli_bin,
    ),
    Field(
        name="task_manager_bin",
        type=Path,
        default=None,
        help="Path to gqe_task_manager binary",
        cli_flags=("--task-manager-bin",),
        resolve=_resolve_task_manager_bin,
    ),
    Field(
        name="num_gpus",
        type=int,
        default=1,
        help="Number of GPUs / task managers",
        cli_flags=("--num-gpus",),
    ),
    Field(
        name="swept_sqlite",
        type=Path,
        default=None,
        help="Prior sweep .db3 file or folder (pretuned mode)",
        cli_flags=("--swept-sqlite",),
    ),
    Field(
        name="log_level",
        type=str,
        default="INFO",
        help="Logging verbosity: DEBUG, INFO, WARNING, ERROR, QUIET",
        cli_flags=("--log-level",),
    ),
    Field(
        name="time_breakdown",
        type=bool,
        default=False,
        help="Capture CUPTI activity breakdown (populates gqe_run_time_breakdown)",
        cli_flags=("--time-breakdown",),
    ),
    Field(
        name="cupti_metrics",
        type=list,
        default=[],
        help="CUPTI range metrics to profile",
        cli_flags=("--cupti-metrics",),
    ),
    Field(
        name="nsys",
        type=Path,
        default=None,
        help=(
            "Profile the engine under Nsight Systems, one report per experiment. "
            "Takes an optional path to the nsys binary; bare --nsys resolves it from PATH. "
            "Disables the gqe_bench NVTX plugin, so no timing rows are written and "
            "--time-breakdown and --cupti-metrics have no effect"
        ),
        cli_flags=("--nsys",),
        nargs="?",
        # Value taken when the flag is given without one: resolved via PATH.
        const="nsys",
    ),
    Field(
        name="nsys_trace",
        type=str,
        default=None,
        help="APIs to trace, comma-separated, e.g. nccl,cuda,nvtx. Requires --nsys",
        cli_flags=("--nsys-trace",),
        requires="nsys",
        resolve=_resolve_nsys_trace,
    ),
    Field(
        name="nsys_launch_args",
        type=str,
        default=None,
        help=(
            "Launch flags as a single string, e.g. --cuda-memory-usage=true. Use the "
            "equals form, since a value starting with '-' is otherwise read as a flag: "
            "--nsys-launch-args=--cuda-memory-usage=true. nsys splits its flags across "
            "verbs; these go to `nsys launch`, --nsys-args to `nsys start`. The trace "
            "flag is reserved by gqe_bench. Requires --nsys"
        ),
        cli_flags=("--nsys-launch-args",),
        requires="nsys",
        resolve=_resolve_nsys_launch_args,
    ),
    Field(
        name="nsys_args",
        type=str,
        default=None,
        help=(
            "Collection flags as a single string, e.g. --sample, --cpuctxsw, "
            "--gpu-metrics-*. Use the equals form, since a value starting with '-' is "
            "otherwise read as a flag: --nsys-args=--gpu-metrics-set=gb10x. "
            "The output flag is reserved by gqe_bench. Requires --nsys"
        ),
        cli_flags=("--nsys-args",),
        requires="nsys",
        resolve=_resolve_nsys_args,
    ),
    Field(
        name="nsys_prefix",
        type=str,
        default=None,
        help=(
            "Filename prefix for profiles, named "
            "{prefix}_Q{query}_{decimal_type}_e{experiment}_{n}.nsys-rep. Omitted, reports "
            "are named Q{query}_{decimal_type}_e{experiment}_{n}. Requires --nsys"
        ),
        cli_flags=("--nsys-prefix",),
        requires="nsys",
        resolve=_resolve_nsys_prefix,
    ),
    Field(
        name="nsys_output_dir",
        type=Path,
        default=None,
        help="Directory to write .nsys-rep files to (default: cwd). Requires --nsys",
        cli_flags=("--nsys-output-dir",),
        requires="nsys",
        resolve=_resolve_nsys_output_dir,
    ),
    # Sweep fields that can't be auto-derived from DDL
    Field(
        name="zone_map_partition_size",
        type=list,
        default=[200000],
        help="Zone map partition size. Only takes effect with use_partition_pruning.",
        mode=FieldMode.SWEEP,
        cli_flags=("--zone-map-partition-size",),
        element_type=_cli_converter(DATA_INFO_MAPPING, "zone_map_partition_size"),
    ),
    Field(
        name="storage_device_kind",
        type=list,
        default=["shared_segment_memory"],
        help="Storage device kind",
        mode=FieldMode.SWEEP,
        cli_flags=("--storage-device-kind",),
    ),
    # A cross-table override field, like storage_device_kind above: its column
    # lives in data_info rather than a gqe_* table, so the default is supplied
    # here rather than by the DDL.
    Field(
        name="decimal_type",
        type=list,
        default=["double"],
        help="Representation for DECIMAL columns: double (FLOAT64) or decimal (DECIMAL64, scale -2)",
        mode=FieldMode.SWEEP,
        cli_flags=("--decimal-type",),
    ),
    Field(
        name="query_source",
        type=list,
        default=[QuerySource.SQL],
        help="Query source",
        mode=FieldMode.SHARED,
        cli_flags=("--query-source",),
    ),
    Field(
        name="load_all_data",
        type=bool,
        default=False,
        help="Package all queries per DataInfo into one load (true) vs reload per query (false).",
        mode=FieldMode.SHARED,
        cli_flags=("--load-all-data",),
    ),
    Field(
        name="serialize_to_disk",
        type=bool,
        default=False,
        help="After loading each table into memory, serialize it to disk under "
        "<dataset>/<table>/serialized_data/ (requires --load-all-data).",
        mode=FieldMode.SHARED,
        cli_flags=("--serialize-to-disk",),
    ),
    Field(
        name="deserialize_from_disk",
        type=bool,
        default=False,
        help="Load each table from its on-disk serialized snapshot instead of from parquet.",
        mode=FieldMode.SHARED,
        cli_flags=("--deserialize-from-disk",),
    ),
)

FIELDS: tuple[Field, ...] = _MANUAL_FIELDS + _auto_fields(
    DATA_INFO_MAPPING,
    QUERY_PARAMS_MAPPING,
    explicit=frozenset(f.name for f in _MANUAL_FIELDS),
)

QUERY_FIELDS: frozenset[str] = frozenset(QUERY_PARAMS_MAPPING.sweep_fields)

_FIELDS_BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}
_VALID_NAMES: frozenset[str] = frozenset(f.name for f in FIELDS) | {"query_overrides"}


def _fresh_defaults() -> dict[str, Any]:
    """Build a defaults dict with deep-copied values, one fresh snapshot per call."""
    return {f.name: f.get_default() for f in FIELDS}


def _resolve_defaults(ns: Namespace) -> None:
    """Call resolve callbacks for fields that are still None after parsing."""
    for field in FIELDS:
        if field.resolve is not None and getattr(ns, field.name, None) is None:
            setattr(ns, field.name, field.resolve(ns))


def _warn_missing_binaries(ns: Namespace) -> None:
    """Warn if binary fields are None after resolution (likely execution will fail).

    Fields that require another flag (``requires``), like ``nsys_output_dir``,
    are skipped by the ``field.requires is None`` check: their resolver returns
    None when that flag is not set, not because a binary is missing.
    """
    for field in FIELDS:
        if (
            field.resolve is not None
            and field.type is Path
            and field.requires is None
            and getattr(ns, field.name, None) is None
        ):
            logger.warning(
                "%s not found in the installed gqe-bench package. "
                "Provide --%s explicitly, or reinstall the package to build it.",
                field.name,
                field.name.replace("_", "-"),
            )


def _coerce_paths(ns: Namespace) -> None:
    """Coerce string values to Path for Path-typed fields (needed for JSON config)."""
    for field in FIELDS:
        if field.type is Path:
            val = getattr(ns, field.name, None)
            if isinstance(val, str):
                setattr(ns, field.name, Path(val))


def _validate_dataset(ns: Namespace) -> None:
    """Reject a run with no dataset directory, from CLI flags or a config file alike."""
    if ns.dataset is None:
        raise ValueError("dataset is required")


def _validate_pretuned_args(ns: Namespace) -> None:
    """Raise ValueError if sweep-only args are provided alongside --swept-sqlite."""
    if not ns.swept_sqlite:
        return
    for field in FIELDS:
        if field.mode != FieldMode.SWEEP:
            continue
        val = getattr(ns, field.name, None)
        if val is not None and val != field.get_default():
            raise ValueError(
                f"--{field.name.replace('_', '-')} cannot be used with --swept-sqlite "
                f"(sweep dimensions come from the DB in pretuned mode)"
            )


def _validate_log_level(ns: Namespace) -> None:
    """Normalize log_level to upper-case and reject unknown levels."""
    level = str(ns.log_level).upper()
    if level not in LOG_LEVELS:
        valid = ", ".join(LOG_LEVELS)
        raise ValueError(f"Invalid log level {ns.log_level!r}; choose from: {valid}")
    ns.log_level = level


def _validate_execution_backend(ns: Namespace) -> None:
    """Reject malformed or unknown execution-backend sweep values."""
    configured = getattr(ns, "execution_backend", [])
    if not isinstance(configured, list):
        raise ValueError("execution_backend must be a list")

    values = list(configured)
    for override in getattr(ns, "query_overrides", []) or []:
        override_values = override.get("execution_backend", [])
        if not isinstance(override_values, list):
            raise ValueError("query_overrides execution_backend must be a list")
        values.extend(override_values)

    if any(not isinstance(value, str) for value in values):
        raise ValueError("execution_backend values must be strings")
    invalid = sorted({value for value in values if value not in EXECUTION_BACKENDS})
    if invalid:
        valid = ", ".join(sorted(EXECUTION_BACKENDS))
        raise ValueError(f"Invalid execution backend(s) {invalid}; choose from: {valid}")


def _validate_serialize(ns: Namespace) -> None:
    """Reject --serialize-to-disk/--deserialize-from-disk combinations the engine does not support.

    Serialization writes a full-table snapshot, which requires every table to be loaded whole
    (--load-all-data). The two flags are also mutually exclusive within one run.

    Secondary compression format is not supported when serializing to disk.
    """
    if getattr(ns, "serialize_to_disk", False) and getattr(ns, "deserialize_from_disk", False):
        raise ValueError("--serialize-to-disk and --deserialize-from-disk are mutually exclusive")
    if not getattr(ns, "serialize_to_disk", False):
        return
    if not getattr(ns, "load_all_data", False):
        raise ValueError("--serialize-to-disk requires --load-all-data")
    secondary = getattr(ns, "secondary_compression_format", None) or []
    unsupported = sorted({str(value) for value in secondary if str(value).lower() != "none"})
    if unsupported:
        raise ValueError(
            "--serialize-to-disk requires secondary_compression_format to be 'none'; got: "
            + ", ".join(unsupported)
        )


def _validate_requires(ns: Namespace) -> None:
    """Reject fields whose ``requires`` field is unset."""
    for field in FIELDS:
        if field.requires is None:
            continue
        val = getattr(ns, field.name, None)
        if val is None:
            continue
        if getattr(ns, field.requires, None) is None:
            raise ValueError(
                f"--{field.name.replace('_', '-')} requires "
                f"--{field.requires.replace('_', '-')}"
            )


def _validate_nsys(ns: Namespace) -> None:
    """Resolve --nsys to an executable and check the nsys arguments.

    Raises ValueError if nsys is not found, if --nsys-args or --nsys-launch-args
    contains a flag gqe_bench sets itself (output path, session, trace list), or
    if nsys rejects the arguments.
    """
    if ns.nsys is None:
        return

    # Handles both the bare-flag sentinel and an explicit path: which() returns
    # a path unchanged when it is already an executable file.
    resolved = shutil.which(str(ns.nsys))
    if resolved is None:
        raise ValueError(
            f"nsys not found: {str(ns.nsys)!r}. Pass --nsys <path> or put nsys on PATH"
        )
    ns.nsys = Path(resolved)

    # gqe_bench owns the output path so it can name each capture after the
    # experiment that produced it; a user-supplied -o would break that mapping.
    # Catches the attached "-o<path>" spelling too, not just "-o" alone.
    for token in shlex.split(ns.nsys_args):
        if token.startswith("-o") or token == "--output" or token.startswith("--output="):
            raise ValueError(
                f"--nsys-args may not contain {token!r}; gqe_bench owns the nsys output path"
            )
        if token.startswith("--session"):
            raise ValueError(
                f"--nsys-args may not contain {token!r}; gqe_bench owns the nsys session"
            )

    # --nsys-trace owns the trace list and gqe_bench owns the session, both of
    # which are already on the launch verb.
    for token in shlex.split(ns.nsys_launch_args):
        if token in ("-t", "--trace") or token.startswith("--trace="):
            raise ValueError(f"--nsys-launch-args may not contain {token!r}; use --nsys-trace")
        if token.startswith("--session"):
            raise ValueError(
                f"--nsys-launch-args may not contain {token!r}; gqe_bench owns the nsys session"
            )

    try:
        NsysProfiler.verify_args(ns.nsys, ns.nsys_trace, ns.nsys_launch_args, ns.nsys_args)
    except NsysError as e:
        raise ValueError(f"nsys rejected these arguments: {e.stderr}") from e


def load_config(path: str) -> dict:
    """Load and validate a JSON5 configuration file.

    Raises ValueError if the config contains keys not in the FIELDS schema.
    """
    with open(path, "r") as f:
        config = json5.load(f)

    unknown = [key for key in config if key not in _VALID_NAMES]
    if unknown:
        raise ValueError(f"Unknown fields in config: {unknown}")

    return config


def _coerce_float_values(values: dict) -> None:
    """Convert JSON integers in float-typed sweep lists to floats, in place.

    gqe-cli types a SET value by its SQL literal, and GQE rejects an integer for a double option.
    """
    for mapping in (DATA_INFO_MAPPING, QUERY_PARAMS_MAPPING):
        for name in mapping.sweep_fields:
            entries = values.get(name)
            if isinstance(entries, list) and mapping.field_py_type(name) is float:
                values[name] = [
                    float(v) if isinstance(v, int) and not isinstance(v, bool) else v
                    for v in entries
                ]


def config_to_args(config: dict) -> Namespace:
    """Convert a JSON config dict to an argparse Namespace, filling defaults."""
    args_dict = _fresh_defaults()
    for key, value in config.items():
        args_dict[key] = value
    _coerce_float_values(args_dict)
    for override in args_dict.get("query_overrides") or []:
        _coerce_float_values(override)
    ns = Namespace(**args_dict)
    _coerce_paths(ns)
    return ns


def _check_cli_overrides(argv: list[str] | None) -> None:
    """Warn if CLI args other than --json/--swept-sqlite are provided with --json."""
    if argv is None:
        argv = sys.argv[1:]
    allowed = {"--json", "-j", "--swept-sqlite"}
    i = 0
    other_args = []
    while i < len(argv):
        arg = argv[i]
        if arg in allowed:
            i += 2
            continue
        if arg.startswith("-"):
            other_args.append(arg)
        i += 1
    if other_args:
        logger.warning(
            "The following CLI arguments are ignored when using --json: %s",
            other_args,
        )


def build_arg_parser() -> ArgumentParser:
    """Build a CLI argument parser driven by FIELDS."""
    parser = ArgumentParser(description="GQE benchmark runner")

    parser.add_argument(
        "--json",
        "-j",
        metavar="CONFIG",
        help="Path to a JSON5 config file (overrides all other args)",
    )

    for field in FIELDS:
        default = field.get_default()
        if field.type is bool:
            parser.add_argument(
                *field.cli_flags, action="store_true", default=default, help=field.help
            )
        elif field.type is list:
            parser.add_argument(
                *field.cli_flags,
                nargs="+",
                type=field.element_type,
                default=default,
                help=field.help,
            )
        else:
            parser.add_argument(
                *field.cli_flags,
                type=field.type,
                nargs=field.nargs,
                const=field.const,
                default=default,
                help=field.help,
            )

    return parser


def parse_args(argv: list[str] | None = None) -> Namespace:
    """Parse CLI args or JSON5 config into a unified Namespace.

    Modes:
      - Sweep: --json config.json5 or CLI args with sweep dimensions.
      - Pretuned: --swept-sqlite db.db3. Reads params from prior sweep DB.
      - Combined: --json config.json5 --swept-sqlite db.db3. JSON provides
        shared config; sweep dimensions in JSON are ignored.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.json:
        _check_cli_overrides(argv)
        config = load_config(args.json)
        ns = config_to_args(config)
        if args.swept_sqlite:
            ns.swept_sqlite = Path(args.swept_sqlite)
    else:
        result = _fresh_defaults()
        for key in vars(args):
            if key == "json":
                continue
            val = getattr(args, key)
            if val is not None or key not in result:
                result[key] = val
        ns = Namespace(**result)

    _validate_dataset(ns)
    _validate_pretuned_args(ns)
    _validate_requires(ns)
    _validate_log_level(ns)
    _validate_execution_backend(ns)
    _validate_serialize(ns)
    _resolve_defaults(ns)
    _validate_nsys(ns)
    _warn_missing_binaries(ns)
    return ns


def get_query_execution_params(args: Namespace, query_name: str) -> dict[str, list]:
    """Get query-level sweep params, merging any per-query overrides."""
    overrides = getattr(args, "query_overrides", [])
    matching = [entry for entry in overrides if query_name in entry.get("queries", [])]

    params: dict[str, list] = {}
    for field_name in QUERY_FIELDS:
        override_values = [entry[field_name] for entry in matching if field_name in entry]
        if override_values:
            merged = list(override_values[0])
            for values in override_values[1:]:
                seen = set(merged)
                merged.extend(v for v in values if v not in seen)
            params[field_name] = merged
        else:
            params[field_name] = getattr(args, field_name)
    return params
