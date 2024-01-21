Expression
==========

.. automodule:: gqe.expression

Constructing expressions
------------------------

In most cases, the first expression we construct is a
:class:`column reference expression <gqe.expression.ColumnReference>`, which returns a column of
the associated relation. Another commonly-used leaf node of the expression is the
:class:`literal expression <gqe.expression.Literal>`, which represents literals like `"Brand#23"`
or `15.0`.

Once we have some basic expressions, we can use
:class:`binary operators <gqe.expression.BinaryOpExpression>` to construct more complicated
expressions. The following table shows the supported operators.

==============  ===================
Operator          Shorthand
==============  ===================
AndExpr          `expr1 & expr2`
EqualExpr        `expr1 = expr2`
LessExpr         `expr1 < expr2`
MultiplyExpr     `expr1 * expr2`
DivideExpr       `expr1 / expr2`
==============  ===================

Note that GQE also provides shorthands to make constructing these expressions easier. For example,

.. code-block:: python

    (CR(1) == Literal("Brand#23")) & (CR(2) == Literal("MED BOX"))

is equivalent to

.. code-block:: python

    AndExpr(EqualExpr(CR(1), Literal("Brand#23")), EqualExpr(CR(2), Literal("MED BOX")))


API references
--------------

.. autoclass:: gqe.expression.Expression

.. autoclass:: gqe.expression.ColumnReference
    :special-members: __init__

.. autoclass:: gqe.expression.Literal
    :special-members: __init__

.. autoclass:: gqe.expression.BinaryOpExpression
    :special-members: __init__
