# Motivation, or: Why should I use this strange tool?

This motivation is divided into two parts: a story that motivates a better
workflow, and concrete reasons why other proposed alternatives don't work well.

## A Story of the Failed Workflow

Jon Snow is a very proficient DevTech. He diligently produces benchmark numbers
to show his "crow" colleagues.

### Plotting

One day, he runs some well-trodden code on ComputeLab. He collects the numbers
into a spreadsheet along with the grid search parameters, and wants to write a
plot script.

The grid search has a lot of combinations, and it took him some time to collect
the results. As he's plotting, he discovers that a logarithmic axis would be
nicer than a linear axis, but that would mean adjusting the parameter scaling
from linear to exponential.

That done, he's uncertain if his results are accurate and precise. He measured
only a single run, and can't compute standard deviation, nor a mean and median.

He writes a shell script that logs the the parameters, and extracts the results
from `stdout`. He stores this information in a CSV file.

After more runs, he is sitting on 100 megabytes of data. Opening the CSV in
Excel becomes painfully slow. He doesn't even need all of the parameters for his
plot. If only there was a tool to quickly filter and shape his data!

In the meantime, he switched from Excel to Seaborn in Python. After resolving
some CSV import issues due to a stray `,` and a number column mapped to a string
type instead of an integer, he finally generates the plot.

### Sharing and Documenting

His numbers are great and his plot makes some waves. Sam Tarly, Jon's colleague,
wants to use Jon's numbers as a baseline in their own plot. Jon is happy to
share and sends over his 100 MB CSV.

Sam isn't amused. First, he needs to download a large file. Second, he doesn't
know how to interpret Jon's numbers. Is that performance number in seconds or in
milliseconds, he wonders. What does "column 1" even represent? It doesn't help
that data aren't clean, e.g., with some entries spelled "white walker" and
others "Whit Walker".

Sam wishfully thinks: "If only the data were in multiple tables. Then, it would
be smaller AND more consistent, as there would be a single entry per parameter
value (i.e., "white walker" and "watchman") referenced by a foreign key. Better
still if we could agree on with column names I'm familiar with."

Unfortunately, Jon is on patrol (i.e., "out-of-office") and not easily
reachable.

### Debugging

When Jon comes back from patrol, he reruns the code on ComputeLab, on the exact
same node he's used before. But this time, the results are slower than he
previously experienced. His mind clouds with doubts:

- Did the CUDA version change on this machine? He knows the admins updated it
  recently, but isn't sure if it before or after his last benchmark run.
- Is there something wrong with the GPU clock rate, maybe it's overheating due
  to a hardware issue?
- Did he parameterize his code differently last time?
- Is it really the same node he remembers? After all, it's been a while.
- Did he note down the correct numbers last time?

Unfortunately, he had manually copy-pasted the numbers into a spreadsheet, with
just enough context to show them around. He doesn't remember the exact details
that might help him now.

### Summary

Jon and Sam hold a meeting and come up with the following plan:

 - They will create a typed schema with agreed-upon column names that documents
   the parameters and results.
 - The schema will also describe details about the experiment setup that might
   be relevant later for debugging or plotting.
 - The schema will be
   [normalized](https://en.wikipedia.org/wiki/Database_normalization#Normal_forms),
   to [2NF](https://en.wikipedia.org/wiki/Second_normal_form#Example), possibly
   even
   [3NF](https://en.wikipedia.org/wiki/Third_normal_form#%22Nothing_but_the_key%22).
   This results in multiple tables.
 - To store multiple tables, they know they need referential integrity for
   foreign keys. That points towards using a database. But they want small
   files, not a remote ODBC or JDBC connection.
 - Sam suggests [SQLite](https://www.sqlite.org/about.html). It defines a file
   format, and can be embedded as a library into other tools if needed.
 - Jon agrees to SQLite, because the format is supported by
   [Python](https://docs.python.org/3/library/sqlite3.html),
   [R](https://rsqlite.r-dbi.org/), [C/C++](https://sqlite.org/cintro.html), and
   even other databases like
   [DuckDB](https://duckdb.org/docs/extensions/sqlite.html). This is perfect for
   plotting.
 - The file will be smaller than before. This is because normalizing
   deduplicates strings, numbers are stored as binary types instead of strings,
   and [SQLite bit-packs integers](https://sqlite.org/fileformat2.html#varint).
 - They will collect data either with a shell script as before, or by directly
   inserting rows from their code. The latter would give them ACID guarantees,
   in case their program crashes or multiple threads insert in parallel.
 - SQL allows them to quickly filter and shape their data as needed.

In summary, the schema implemented as SQLite will allow them to analyze data
more effectively.

## Non-Alternatives

Commonly-used tools lead to inhospital workflows. This is a (incomplete) list of
examples.

### Unstructured data (raw log files, etc.)

This has a single advantage: it (hopefully) includes *all* output of the code.

But this isn't enough, as it doesn't include how the code was built (e.g., Git
revision, compiler flags), on which hardware it was run, *when* the experiment
was conducted, etc.

It has serious drawbacks as a data format:

 - Hard to interpret and make sense of. Especially when looking at older logs
   after code changes, or unfamiliar with the code.
 - Hard to consistently extract relevant data from. `grep` and `awk` are great,
   but "homonyms" (unrelated lines that happen to use the same filter keyword)
   and other mistakes can be hard to avoid in large code bases.
 - Size.

Unstructured data are hard to avoid. However, the responsibilty to extract and
structure the *useful* data is with the person conducting the experiment, *at
the time when conducting the experiment*!

### Text-based data formats (CSV, JSON, XML, etc.)

Text files can be good for ad hoc data collection. However, they have drawbacks:

 - Space-inefficient and slow to parse compared to a binary format.
 - No well-defined, consistent schema (XML is an exception if it includes a DSD).
 - No data types (looking at you, CSV).
 - Cannot normalize data within a single file.
 - Not efficiently searchable and editable (try opening a 100 MB file in a text
   editor; if that works well, try modifying a single character and then
   saving).

### Apache Parquet

Much better than text-based formats. Still has some similar drawbacks:

 - No well-defined, consistent database schema (only single files; unless adding
   external tooling, e.g., [Unity
   Catalog](https://github.com/unitycatalog/unitycatalog)).
 - Not a single file when data are normalized.
 - Not as well integrated into non-big data tools as SQLite (e.g., Python
   includes SQLite in it's standard library; every Linux distro ships SQLite).
 - No ACID guarantees (unless adding external tooling, e.g., [Delta
   Lake](https://docs.databricks.com/en/delta/index.html) or [Apache
   Iceberg](https://iceberg.apache.org)). More a nice-to-have than an actual
   requirement, though.

### RDBMS servers (PostgreSQL, MySQL, etc.)

The main issue is that a DBMS server is not a file format. That has drawbacks:

 - Hosting. On what machine to persistently run the server? Maintenance &
   updates responsibility? Security due to open port? Not accessible due to
   firewall / network issue?
 - Schema evolution. SQLite is fire-and-forget, old files have an old schema. In
   a server, schema changes require updating old data to a new schema.
 - Sharing. Easier to send or upload a file than to give access to the server.
   If we extract data from the server, we're back to square one: which file
   format?
