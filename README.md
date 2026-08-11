# GPU Query Executor (GQE) benchmarking suite

This repository contains the accompanying benchmarking suite for the GPU Query Executor (GQE). In addition it also contains scripts for generating TPC-H dataset, and reference files.

## Installation
The commands below assume that the working directory is the local GQE Bench repo, and the GQE Conda environment is active.

```
uv pip install --system -e .
```

By default the package uses upstream GQE from gitlab.

You can pass your own GQE git repository and tag to the install command.

```
uv pip install --system -e . -C cmake.define.GQE_GIT_REPOSITORY=<your_repository> -C cmake.define.GQE_GIT_TAG=<your_tag>
```

You can also pass the path to a local GQE clone to the install command.

```
uv pip install --system -e . -C cmake.define.GQE_SOURCE_DIR=<path_to_local_gqe>
```

You can also use a custom gqe-nvcomp branch/tag to use

```
uv pip install --system -e . -C cmake.define.GQE_NVCOMP_TAG=<gqe-nvcomp commit/branch>
```

Or you could also point it to a local nvcomp directory (This takes precedence over the git tag)

```
uv pip install --system -e . -C cmake.define.GQE_NVCOMP_SOURCE_DIR=<path_to_nvcomp_folder>
```

## Benchmarking

Primarily used for benchmarking TPC-H queries using GQE, but can run custom queries on custom dataset as well.

Basic command for running the benchmark after [installation](#installation):

```bash
gqe-bench --dataset <dataset> --sql <sql_dir> --solution <reference_dir> --output results.db3
```

`gqe-bench` is a console entry point installed with the package; `python -m gqe_bench.runner` is the equivalent module form. All options are documented in the help section:
```bash
gqe-bench -h
```

### JSON5 Configuration

Instead of passing all parameters via command line, you can use a JSON5 config file:

```bash
gqe-bench --json config.json5
```

See `config_templates/tpch_sweep_CI.json5` for the canonical, tested example. A trimmed config:

```json5
{
  "dataset": "/path/to/tpch/sf100",
  "sql": "/path/to/tpch/sql",
  "solution": "/path/to/solutions",
  "output": "results.db3",
  "queries": ["1", "2", "3", "2_fused_filter", "3_fused_filter", "11"],
  "query_source": ["sql", "handcoded"],
  "num_partitions": [1, 2, 4, 8],
  "num_workers": [1],
  "repeat": 6,
  "storage_device_kind": ["boost_shared_memory"],

  "query_overrides": [
    {
      "queries": ["2_fused_filter", "3_fused_filter"],
      "num_partitions": [4, 8],
      "join_use_perfect_hash": [true]
    },
    {
      "queries": ["11"],
      "num_partitions": [1, 2]
    }
  ]
}
```

The `query_overrides` section is a list of override objects. Each object has a `queries` field containing a list of query strings (matched exactly) that the overrides apply to. When an override specifies a parameter, it replaces the global config value for that query. Parameters not specified in any override inherit from the global config. If a query matches multiple override entries, their values are merged (keeping unique values).

Override-able parameters: `num_partitions`, `num_workers`, `use_overlap_mtx`, `join_use_hash_map_cache`, `read_use_zero_copy`, `join_use_unique_keys`, `join_use_perfect_hash`, `join_use_mark_join`, `use_partition_pruning`, `filter_use_like_shift_and`, `aggregation_use_perfect_hash`, `use_ast_jit`.

Validation runs when `solution` (a reference-results directory) is set; omit it to skip validation.

When using `--json`, all other CLI arguments are ignored (a warning is printed if any are provided).

### Query Sources

Each query runs from one of two sources, selected by `query_source` (a list, so a single run can cover both):

- **`sql`** — the query is executed from SQL. For TPC-H the SQL is generated on demand via DuckDB (all 22 queries). Pass `--sql <dir>` to supply your own `q<N>.sql` files instead.
- **`handcoded`** — the query is executed from a hand-built physical plan assembled in-process via the plan DSL (`src/gqe_bench/physical_plan/`). TPC-H ships handcoded plans for most of the suite, including fused-filter variants such as `2_fused_filter`.

When both sources are listed, each query runs under every source it supports and is recorded separately.

### Pretuned Mode

A sweep explores a grid of parameters; pretuned mode replays only the best parameter set per query, read from a prior sweep's `.db3`:

```bash
gqe-bench --dataset <dataset> --solution <reference_dir> --swept-sqlite prior_sweep.db3 --output pretuned.db3
```

The mode is selected by `--swept-sqlite`, which accepts a single `.db3` or a directory of them. Sweep dimensions (partition counts, optimization flags, etc.) come from the DB and cannot be set on the command line in this mode; `--queries` and `--query-source` still filter which results are replayed.

### Output

Results are written to a SQLite `.db3` at the `--output` path: one `experiment` per (query, parameter set), a `run` row per repetition, and a `failed_run` row for any run that errors or fails validation. GQE-specific views summarize the sweep — `gqe_best_parameters` (lowest average duration per query), `gqe_run_parameters`, `failed_experiments`, and others. The full schema is in `src/gqe_bench/sql/system_under_test.sql`.

The `build_info` row identifies the GQE engine that produced the results — commit, branch, and whether the checkout had uncommitted changes — so a `.db3` can be traced back to the engine build it came from.

# Analysis scripts

### Usage
```bash
python scripts/nsys_analysis.py <tool> <sqlite> "<nvtx_range_glob>" [options]
```
- **tool**: `kernel` | `io`
- **sqlite**: path to `.sqlite` exported by Nsight Systems
- **nvtx_range_glob**: SQLite GLOB pattern for NVTX range label (use `*` and `?`, e.g., "*Run Q13*")

### Options
- Common:
  - **-o, --output <file>**: write CSV instead of printing rows
- Kernel tool (`kernel`):
  - **--analysis_type**: `kernel_time_sum` | `kernel_time_effective` (default: `kernel_time_effective`)
  - **--exclude_kernel_glob <glob>**: exclude kernels by demangled name (e.g., "*fused_concatenate*")
- IO tool (`io`):
  - **--analysis_type**: `htod_copy_time_sum` | `htod_copy_size` | `read_time_effective` | `hw_decompress_time_sum` (default: `read_time_effective`)

### Examples
```bash
# Kernel: effective (end-to-end) kernel time
python scripts/nsys_analysis.py kernel --analysis_type kernel_time_effective /path/to/trace.sqlite "*Run Q13*" -o kernel_effective.csv

# Kernel: total kernel time excluding a kernel pattern
python scripts/nsys_analysis.py kernel --analysis_type kernel_time_sum --exclude_kernel_glob "*fused_concatenate*" /path/to/trace.sqlite "*Run Q13*"

# IO: total HtoD copy time from pinned host memory within the NVTX range
python scripts/nsys_analysis.py io --analysis_type htod_copy_time_sum /path/to/trace.sqlite "*Run Q13*"  -o io_htod_time.csv

# IO: effective in-memory read task time (GQE NVTX ranges merged)
python scripts/nsys_analysis.py io --analysis_type read_time_effective /path/to/trace.sqlite "*Run Q13*"

# IO: total decompression engine decompress time
python scripts/nsys_analysis.py io --analysis_type hw_decompress_time_sum /path/to/trace.sqlite "*Run Q2*"
```

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code style and quality. The hooks include:
- **Ruff** for Python linting and formatting (replaces flake8, isort, black, autoflake)
- **clang-format** for C++/CUDA formatting

### Setup

Install the pre-commit hooks (one-time setup):

```bash
pre-commit install
```

### Usage

The hooks will run automatically on `git commit`. To run manually on all files:

```bash
pre-commit run --all-files
```

To run a specific hook:

```bash
pre-commit run ruff --all-files        # linting with auto-fix
pre-commit run ruff-format --all-files # formatting
pre-commit run clang-format --all-files
```

## Data generation and validation generation scripts

### Dependencies
- Python3 interpreter
- DuckDB Python module, e.g., `conda install duckdb`

To generate solution files for queries you can use the `scripts/generate_validation_files.py`.

```bash
usage: python scripts/generate_validation_files.py [dataset] [queries_sql] [results]
```

This will generate `<query>.parquet` in `<output_directory>` for each `<query>.sql` in `<sql_file_or_dir>`


To generate TPC-H dataset:

```bash
usage: python generate_parquet_dataset.py [-h] [-t TMP] [-o OUTPUT] [-s SCALE] [-c CHUNKSIZE]

options:
  -h, --help            Show this help message and exit
  -t TMP, --tmp TMP
                        Override path for temporary DB on disk
  -o OUTPUT, --output OUTPUT
                        Output path (default: ./data)
  -s SCALE, --scale SCALE
                        Scale factor (default: 1)
  -c CHUNKSIZE, --chunksize CHUNKSIZE
                        Max number of rows in each Parquet file (may be rounded up to multiple of 2048, default: 16000000)
  -r SF_PER_CHILD, --sf_per_child SF_PER_CHILD
                        Ratio of scale factor to number of children; decrease to reduce memory footprint (default: 10)
```

## Timing and profiling plugin

Per-run measurements in the `.db3` are produced in-process by `libgqe_bench.so`, an `LD_PRELOAD`'d shared library built from `src/gqe_bench/nvtx_plugin/` when the `GQE_BUILD_NVTX_PLUGIN` CMake option is on. The runner sets `LD_PRELOAD` itself; no manual export is needed.

The library is loaded into more than one process, so it activates only inside `gqe_task_manager`. Once active it observes the NVTX ranges GQE emits during plan execution: an outer `execute_plan` range, and one range per pipeline stage inside it.

### What it records

The plugin captures in three modes, of which the two CUPTI modes are independent opt-ins:

| Mode | Enabled by | Populates |
|---|---|---|
| Stage timing | always on | `run`, plus one aggregated `gqe_run_ext` row per stage |
| CUPTI Activity | `--time-breakdown` | `gqe_run_time_breakdown`, `gqe_run_cupti_*_activity` |
| CUPTI Range Profiler | `--cupti-metrics <list>` | `gqe_run_ext` counter rows |

The stages are fixed: `build_task_graph`, `execute_task_graph`, and `collect_results`, wrapped by an outer `execute_plan` range. Only the execute stage drives the capturers; the other two are pure timing.

### Multi-GPU aggregation

Ranks coordinate through a shared-memory segment they all attach to before the first run. Each rank publishes its own per-run data there. Rank 0 owns the database connection, gathers every rank's contribution, and commits one transaction per run.

Per-rank rows carry the `gpu_info.g_id` resolved from each device's UUID. Aggregated stage rows use a NULL `gpu_info_id` to mark them as the cross-rank reduction. The `run` table still gets one row per repetition regardless of rank count.

The segment name, rank count, and size arrive by environment variable. An unset name disables the attach.

## Generated constants and files

### Constants shared by the plugin and the runner

The plugin and the Python layer share two sets of constants: the environment-variable names the runner passes to the plugin (SQLite path, shared-memory coordination, CUPTI metric selection), and the per-stage metric names stored in each run's timing breakdown. Both are defined once in `cmake/codegen.cmake` and generated into each side:

| Consumer | Artifact | Source |
|---|---|---|
| C++ plugin | `env.hpp` | `nvtx_plugin/env.hpp.in` |
| C++ plugin | `stages.hpp` | `nvtx_plugin/stages.hpp.in` |
| Python | `gqe_bench/_artifacts.py` | `cmake/codegen.cmake` |

Because both sides come from the same variables, the C++ and Python definitions cannot drift.

To change or add a shared constant, edit `cmake/codegen.cmake` and rebuild. Never edit the generated `env.hpp`, `stages.hpp`, or `_artifacts.py`. The per-constant documentation lives in the two templates.

### Generated files

`cmake/codegen.cmake` produces every file below, and lists them at the top of that file:

| File | Generated | Destination |
|---|---|---|
| `nvtx_plugin/env.hpp` | configure time, plugin on | plugin targets' include path |
| `nvtx_plugin/stages.hpp` | configure time, plugin on | plugin targets' include path |
| `_artifacts.py` | configure time, always | package root |
| `_build_info.py` | build time, engine on (else a sentinel) | package root |
| `_agg_protocol_sizes.py` | build time, plugin on (else a sentinel) | package root |
| `physical_plan/*_pb2.py` | build time, engine on | `gqe_bench/physical_plan/` |

All are written under the build directory and installed into the package from there. Nothing is generated into the source tree.

### Physical-plan protos

The `*_pb2.py` modules are pre-compiled at build time from the `.proto` files in the gqe checkout the engine is built from, so the plans a handcoded query serializes cannot disagree with the format the engine reads. Compiling them here rather than at import time also keeps the protobuf compiler out of every process that imports `gqe_bench`.

Each proto is staged under its Python package path before compilation, with its imports rewritten to match, so the generated modules import each other as `from gqe_bench.physical_plan import data_type_pb2` rather than by bare module name.

The `protobuf` pin in `pyproject.toml` follows the compiler that produced the modules. Generated code rejects a runtime of a different major version, or one older than the compiler.

### NVTX range names come from GQE

The range names the plugin matches, the outer `execute_plan` range and its inner stages, are emitted by GQE itself and appear as literals in `stages.hpp.in`. Renaming one requires a coordinated change on the GQE side, not just a CMake edit.
