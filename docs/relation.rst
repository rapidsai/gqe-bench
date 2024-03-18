Relation
========

.. automodule:: gqe.relation

Constructing relations
----------------------

Typically, the first relation we construct is a
:class:`ReadRelation <gqe.relation.ReadRelation>` to get a table from the catalog. We can use the
:func:`read <gqe.relation.read>` function to achieve this. For example, the following code reads the
`part` table with three columns `p_partkey`, `p_brand` and `p_container` from a catalog named
`catalog`.

.. code-block:: python

   part = read(catalog, "part", ["p_partkey", "p_brand", "p_container"])

Once we have our first relation, we could call each relation's constructor to get new relations.
However, a more elegant way is to use the helper methods in the
:class:`Relation <gqe.relation.Relation>` base class. For example, we could use the
:meth:`filter <gqe.relation.Relation.filter>` method to select a subset of rows. The following code
select rows that column 1 is "Brand#23", and column 2 is "MED BOX".

.. code-block:: python

    from gqe.expression import Literal
    from gqe.expression import ColumnReference as CR

    part = read(catalog, "part", ["p_partkey", "p_brand", "p_container"])
    part = part.filter((CR(1) == Literal("Brand#23")) & (CR(2) == Literal("MED BOX")))

Load a relation from a Substrait file
-------------------------------------

Currently, the Python wrapper does not support the substrait consumer.

API references
--------------

.. autoclass:: gqe.relation.ReadRelation
.. autoclass:: gqe.relation.FilterRelation
.. autoclass:: gqe.relation.JoinRelation
.. autoclass:: gqe.relation.AggregateRelation
.. autoclass:: gqe.relation.ProjectRelation
.. autoclass:: gqe.relation.SortRelation
.. autoclass:: gqe.relation.FetchRelation

.. autofunction:: gqe.relation.read

.. autoclass:: gqe.relation.Relation
    :members:
