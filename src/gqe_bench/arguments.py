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
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import json5

from gqe_bench._artifacts import CLI, NODE_MANAGER, TASK_MANAGER
from gqe_bench.logger import LOG_LEVELS
from gqe_bench.query_source import QuerySource
from gqe_bench.schema import DATA_INFO_MAPPING, QUERY_PARAMS_MAPPING, TableMapping
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
    """

    name: str
    type: type
    default: Any
    help: str
    mode: FieldMode = FieldMode.SHARED
    cli_flags: tuple[str, ...] = ()
    resolve: Callable[[Namespace], Any] | None = None

    def get_default(self) -> Any:
        """Return a deep copy of ``default``.

        Use this rather than reading ``.default`` directly: callers that place
        the result on a Namespace can safely mutate it without affecting other
        Namespaces or this Field's own default.
        """
        return copy.deepcopy(self.default)


# Resolve callbacks for binary discovery.
#
# Fires only when the user didn't supply the corresponding CLI flag (Field's
# resolve contract). Returns the package-relative path if the binary is
# present, None otherwise. Callers that require a path at spawn time convert
# None → ArtifactMissing via Artifact.require() at the spawn site — see
# gqe_bench/session.py.


def _resolve_server_bin(ns: Namespace) -> Path | None:
    """Resolve the gqe_node_manager binary from the installed package."""
    return NODE_MANAGER.locate()


def _resolve_task_manager_bin(ns: Namespace) -> Path | None:
    """Resolve the gqe_task_manager binary from the installed package."""
    return TASK_MANAGER.locate()


def _resolve_cli_bin(ns: Namespace) -> Path | None:
    """Resolve the gqe-cli binary from the installed package."""
    return CLI.locate()


_HELP_OVERRIDES: dict[str, str] = {
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
    # Sweep fields that can't be auto-derived from DDL
    Field(
        name="zone_map_partition_size",
        type=list,
        default=[200000],
        help="Zone map partition size. Only takes effect with use_partition_pruning.",
        mode=FieldMode.SWEEP,
        cli_flags=("--zone-map-partition-size",),
    ),
    Field(
        name="storage_device_kind",
        type=list,
        default=["boost_shared_memory"],
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
    """Warn if binary fields are None after resolution (likely execution will fail)."""
    for field in FIELDS:
        if (
            field.resolve is not None
            and field.type is Path
            and getattr(ns, field.name, None) is None
        ):
            logger.warning(
                "%s not found on PATH. Build with build_flight_sql.sh "
                "or provide --%s explicitly.",
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


def config_to_args(config: dict) -> Namespace:
    """Convert a JSON config dict to an argparse Namespace, filling defaults."""
    args_dict = _fresh_defaults()
    for key, value in config.items():
        args_dict[key] = value
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
            parser.add_argument(*field.cli_flags, nargs="+", default=default, help=field.help)
        else:
            parser.add_argument(*field.cli_flags, type=field.type, default=default, help=field.help)

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
        if ns.dataset is None:
            parser.error("--dataset is required")

    _validate_pretuned_args(ns)
    _validate_log_level(ns)
    _resolve_defaults(ns)
    if ns.dataset is not None or args.json is not None:
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
