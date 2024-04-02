Welcome to GQE's documentation!
===============================

**G**\PU **Q**\uery **E**\xecutor (GQE) is a proof-of-concept query engine for running data
analytics queries on the GPUs. GQE's core backend is developed in C++/CUDA, and this Python layer
is built on top of the C++ APIs. It aims to allow interactive query execution without recompilation.
For example, `benchmark/hardcoded/tpch_q17.py` implements TPC-H Q17.

.. toctree::
   :maxdepth: 2

   install
   relation
   expression
   catalog
   execute
