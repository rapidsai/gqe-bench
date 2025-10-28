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

## Benchmarking

Used for benchmarking TPC-H queries using GQE.

 Instruction for running the benchmark can be found [here](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=DevtechCompute&title=Run+TPC+Benchmarks), in the section "Run TPC-H Queries with Python interface".

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
