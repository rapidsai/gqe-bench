# GPU Query Executor (GQE) python interface

This repository contains the python interface for the GPU Query Executor (GQE). In addition it also contains scripts for generating TPC-H dataset, and reference files. 

## Installation

```
pip install -e .
```

By default the package uses upstream GQE from gitlab.

You can pass your own GQE git repository and tag to the install command.

```
pip install -e . -C cmake.define.GQE_GIT_REPOSITORY=<your_repository> -C cmake.define.GQE_GIT_TAG=<your_tag>
```

You can also pass the path to a local GQE clone to the install command.

```
pip install -e . -C cmake.define.GQE_SOURCE_DIR=<path_to_local_gqe>
```

You can also use a custom gqe-nvcomp branch/tag to use

```
pip install -e . -C cmake.define.GQE_NVCOMP_TAG=<gqe-nvcomp commit/branch>
```

Or you could also point it to a local nvcomp directory (This takes precedence over the git tag)

```
pip install -e . -C cmake.define.GQE_NVCOMP_SOURCE_DIR=<path_to_nvcomp_folder>
```

## Benchmarking

Used for benchmarking TPC-H queries using GQE.

Instruction for running the benchmark can be found [here](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=DevtechCompute&title=Run+TPC+Benchmarks), in the section "Run TPC-H Queries with Python interface".

### JSON Configuration

Instead of passing all parameters via command line, you can use a JSON config file:

```bash
python -m gqe.benchmark.run_tpch_parameter_sweep --json config.json
```

Example `config.json`:

```json
{
  "dataset": "/path/to/tpch/sf100",
  "plan": "/path/to/substrait/plans",
  "solution": "/path/to/solutions/q%d.parquet",
  "output": "results.db",
  "queries": ["1", "2", "3", "2_fused_filter", "3_fused_filter", "11"],
  "partitions": [1, 2, 4, 8],
  "workers": [1],
  "repeat": 6,
  "storage_kind": ["numa_pinned_memory"],
  "query_source": "both",
  "verify_results": true,

  "query_overrides": [
    {
      "queries": ["2_fused_filter", "3_fused_filter"],
      "partitions": [4, 8],
      "join_use_perfect_hash": [true]
    },
    {
      "queries": ["11"],
      "partitions": [1, 2]
    }
  ]
}
```

The `query_overrides` section is a list of override objects. Each object has a `queries` field containing a list of query strings (matched exactly) that the overrides apply to. When an override specifies a parameter, it replaces the global config value for that query. Parameters not specified in any override inherit from the global config. If a query matches multiple override entries, their values are merged (keeping unique values).

Override-able parameters: `partitions`, `workers`, `read_use_filter_pruning`, `read_use_overlap_mtx`, `read_use_zero_copy`, `filter_use_like_shift_and`, `join_use_hash_map_cache`, `join_use_unique_keys`, `join_use_perfect_hash`, `join_use_mark_join`, `aggregation_use_perfect_hash`.

When using `--json`, all other CLI arguments are ignored (a warning is printed if any are provided).

## Substrait Plan Generation

Make sure to compile Substrait-producer in GQE with gqe-python using:

```
pip install -e . -C cmake.define.GQE_ENABLE_SUBSTRAIT_PRODUCER=ON
```

Then you can run the script:
```
python gqe/substrait_producer.py [dataset] [sql_queries] [output]
```


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
  - **--analysis_type**: `htod_copy_time_sum` | `htod_copy_size` | `read_time_effective` (default: `read_time_effective`)

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

## Formatting

To fix formatting of python code:

```
conda install black=25.1.0
black .
```

## Data generation and validation generation scripts

### Dependencies
- Python3 interpreter
- DuckDB Python module, e.g., `conda install duckdb`

To generate solution files for queries you can use the `scripts/duckdb_validation.py`.

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
