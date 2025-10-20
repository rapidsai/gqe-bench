# GPU Query Executor (GQE) python interface

This repository contains the python interface for the GPU Query Executor (GQE).

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

## Usage

Used for benchmarking TPC-H queries using GQE.

 Instruction for running the benchmark can be found [here](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=DevtechCompute&title=Run+TPC+Benchmarks), in the section "Run TPC-H Queries with Python interface".


## Formatting

To fix formatting of python code:

```
conda install black=25.1.0
black .
```
