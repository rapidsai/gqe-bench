# Database Benchmarking Tools

Database benchmarking tools helps to benchmark databases, microbenchmarks, and
related code. It's based on an SQLite schema that collects results, parameters,
and the benchmark environment. The idea is to conduct benchmarks systematically,
in a way that we can effectively create plots, document and share results, and
debug unexpected performance effects.

Alongside the SQLite schema, the tools include a Python library to populate the
SQLite database, and a collection of benchmarking scripts built on top of the
library.

## Motivation

Why should you use this new and strange tool? It was created after previous
experiences with different benchmarking and analysis workflows, which left a bad
flavor in the mouth. It's an attempt to model a better workflow.

Find more details in the [ABOUT file](ABOUT.md).

## Proposed Workflow

The general idea is to replace a text-based format with SQLite, and write a
script to fill the SQLite file:

 - Setup the database benchmarking tools.
 - Write a Python benchmarking script for your use-case. This is split into
   three components:
   - A function that iterates over grid search parameters, calls your
     system-under-test for each parameter combination, and calls the database
     benchmarking library to insert the data into SQLite.
   - A Python file that defines grid parameters.
   - A main Python script that ties these components together. It provides a
     commandline interface, which takes the grid search parameters file, data
     location, and possibly other inputs. It then calls the grid search
     function.
 - Insert your application name into the schema. This is typically done in your
   script, but optionally can be performed externally.
 - Optionally extend the SQLite schema with custom parameters for your workflow.
 - Run the script on ComputeLab or a machine of your choice.
 - Inspect your fresh data ad hoc with the SQLite commandline tool:
   - Directly with: `sqlite3 [your-file]`
   - or [the Python-embedded
     version](https://docs.python.org/3/library/sqlite3.html#command-line-interface)
     with: `python -m sqlite3 [your-file]` (since Python 3.12).
 - Collect the SQLite file(s) via SSH, NFS, or any other means.
 - Archive the SQLite file(s) in a GitLab repository. This is a backup, but also
   allows easy sharing.
 - Analyze and plot the data with a tool of your choice.

However, nobody is forcing anyone to use this library, or even Python. The
provided SQLite schema can be used with any language or tool. [Data can even be
inserted from multiple tools concurrently](https://www.sqlite.org/faq.html#q5).

## Setup

To install the library and provided benchmarking scripts, run:

```sh
git clone https://gitlab-master.nvidia.com/Devtech-Compute/gqe-benchmarking.git
cd gqe-benchmarking
pip install .
```

## Inserting Your Application Name into the Schema

A benchmark benchmarks a system-under-test (see book R. Jain, "The Art of
Computer Systems Performance Analysis"; [slides
here](https://www.cse.wustl.edu/~jain/cse567-08/ftp/k_05aws.pdf)).

Each system-under-test must be inserted into the `sut_info` table. The system
name should be lower-case, and words separated with a single space character.
The system ID should be a constant, globally unique primary key (i.e., unique
across all SQLite files).

Insert this value during benchmark setup. For example, GQE is inserted as:

```sql
INSERT INTO sut_info(s_id, s_name) VALUES (2717457836325482278, 'gqe');
```

To generate a primary key, run: `SELECT abs(random());` in SQLite.

**Hint**: Hand-coded CUDA files can use the pre-defined `cuda` name instead of
inserting a custom system-under-test into `sut_info`.

## Optional: Extending the Schema for Your Application

Most likely, your application has parameters that are specific to it. For
example, GQE defines `MAX_NUM_WORKERS` and other parameters. The schema is
designed to be extensible to store these parameters.

A `X_parameters` table for the system-under-test "X" should be created using a
system-under-test setup file. As an example, see
`scripts/gqe/system_under_test.sql`.

The parameters are read from the file to make the parameters customizable for
each system without modifying this common experiment schema.

The table name should be prefixed by the system name, have a primary key, and
reference `sut_info.s_id`:

```sql
CREATE TABLE gqe_parameters(
  p_id INTEGER PRIMARY KEY,
  p_sut_info_id INTEGER NOT NULL,
  ...,
  FOREIGN KEY (p_sut_info_id) REFERENCES sut_info(s_id)
);
```

Finally, run the SQL file with SQLite on your database file. The library provides support with:

```python
sut_creation_path = importlib.resources.files(
    "database_benchmarking_tools.scripts.your_sut"
).joinpath("system_under_test.sql")
with importlib.resources.as_file(sut_creation_path) as script:
    edb.execute_script(script)
```

Optionally, instead of the library, you can run the SQL file from the commandline:

```sh
sqlite3 the_database_file.db3 < system_under_test.sql
```

## Schema Design Overview

![Experiment Diagram](diagrams/database_experiment_tables.svg)

The SQLite schema consists of 5 tables. These are normalized to be in the [3rd
normal
form](https://en.wikipedia.org/wiki/Third_normal_form#%22Nothing_but_the_key%22),
partially relaxed to
[2NF](https://en.wikipedia.org/wiki/Second_normal_form#Example). As the
candidate key most tables consist of multiple attributes (e.g., the combination
of benchmark parameters), each table includes a surrogate integer key.

Schema extensibility is enabled by an "object-oriented" relation design for
`X_parameters` (see [Naumann lecture, slide
66](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/folien/SS11/DBS_I/DBS1_03_RelationalerEntwurf.pdf)).
This table is a [weak entity](https://en.wikipedia.org/wiki/Weak_entity) of
`sut_info`, i.e., must include the attribute `p_sut_info_id` as part of its
composite primary key. In the diagram, `gqe_parameters` and `spark_parameters`
demonstrate how such a system-specific parameters relation can be designed.

The provided relations are:

 - `experiment`: The "main" relation that contains foreign keys to the other
   relations.
 - `run`: A weak entity of `experiment`, that can be seen as an instantiation of
   the experiment.
 - `build_info`: Describes how the SUT was compiled & built.
 - `hw_info`: Describes the hardware and operating system test bed.
 - `sut_info`: Describes the system-under-test.
