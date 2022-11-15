.. _tql:

Tensor Query Language
=====================

.. role:: sql(code)
    :language: sql

This page describes the Tensor Query Language (TQL), an SQL-like language used for `Querying in Activeloop Platform <https://docs.activeloop.ai/tutorials/querying-datasets>`_
as well as in :meth:`ds.query <deeplake.core.dataset.Dataset.query>` in our Python API. To use queries, install deeplake with ``pip install deeplake[enterprise]``.

Querying datasets is part of our Growth and Enterprise Plan.
    - Users of our Community plan can only perform queries on Activeloop datasets ("hub://activeloop/..." datasets).
    - To run queries on your own datasets, `upgrade your organization's plan <https://www.activeloop.ai/pricing/>`_.

Language
~~~~~~~~

SELECT
------

TQL supports only :sql:`SELECT` statement. Every TQL expression starts with :sql:`SELECT *`. TQL supports only :sql:`*` which means to select all tensors. 
The common syntax for select statement is the following:

.. code-block:: sql

    SELECT * [FROM string] [WHERE expression] [LIMIT number [OFFSET number]] [ORDER BY expression [ASC/DESC]]

Each part of the :sql:`SELECT` statement can be omitted.

:sql:`FROM` expression is allowed, but it does not have any effect on the query, because for now TQL queries are run on a specific dataset, 
so the :sql:`FROM` is known from the context

WHERE
-----

:sql:`WHERE` expression is used to filter the samples in the dataset by conditions. The conditions should be convertible to boolean. 
Any expression which outputs a number will be converted to boolean with non-zero values taken as ``True``. If the expression is not convertible to boolean, 
such as **strings**, **json** objects and **arrays**, the query will print the corresponding error.

ORDER BY
--------

:sql:`ORDER BY` expression orders the output of the query by the given criteria. The criteria can be any expression output of which can be ordered. 
The ordered outputs are either scalar numbers or strings. In addition it can also be json, which contains number or string. 

:sql:`ORDER BY` statement optionally accepts :sql:`ASC/DESC` keywords specifying whether the ordering should be ascending or descending. 
It is ascending by default.

LIMIT OFFSET
------------

:sql:`LIMIT` and :sql:`OFFSET` expressions are used to limit the output of the query by index, as in SQL.  

Expressions
-----------

TQL supports any comparison operator (``==, !=, <, <=, >=``) where the left side is a tensor and the right side is a known value. 

The value can be numeric scalar or array as well as string value. 

String literal should be provided within single quotes (``'``) and can be used on ``class_label``,  ``json`` and ``text`` tensors. 

For class labels it will get corresponding numeric value from the **class_names** list and do numeric comparison. 

For json and text it will do string comparison. The left side of the expression 
can be indexed (subscripted) if the tensor is multidimensional array or json. Jsons support indexing by string, e.g. ``index_meta['id'] == 'some_id'``. 
Jsons can also be indexed by number if the underlying data is array.

Numeric multidimensional tensors can be indexed by numbers, e.g. ``categories[0] == 1`` as well as Python style slicing and 
multidimensional indexing, such as ``boxes[:2]``. This last expression returns array containing the third elements of the initial 
two dimensional array boxes.

TQL supports logical operators - :sql:`AND`, :sql:`OR` and :sql:`NOT`. These operators can be used to combine boolean expressions. 
For example,

.. code-block:: sql

    labels == 0 OR labels == 1

From SQL we also support the following two keywords:

- :sql:`BETWEEN`

.. code-block:: sql

    labels BETWEEN 0 and 5

- :sql:`IN`

.. code-block:: sql

    labels in ARRAY[0, 2, 4, 6, 8]

Functions
---------

There are predefined functions which can be used in :sql:`WHERE` expression as well as in :sql:`ORDER BY` expressions:

- ``CONTAINS`` - checks if the given tensor contains given value - :sql:`CONTAINS(categories, 'person')`
- ``RANDOM`` - returns random number. May be used in :sql:`ORDER BY` to shuffle the output - :sql:`ORDER BY RANDOM()`
- ``SHAPE`` - returns the shape array of the given tensor - ``SHAPE(boxes)``
- ``ALL`` - takes an array of booleans and returns single boolean, ``True`` if all elements of the input array are ``True``
- ``ALL_STRICT`` - same as :sql:`ALL` with one difference. :sql:`ALL` returns ``True`` on empty array, while :sql:`ALL_STRICT` return ``False``
- ``ANY`` - takes an array of booleans and returns single boolean, ``True`` if any of the elements int the input array is ``True``
- ``LOGICAL_AND`` - takes two boolean arrays, does element wise **logical and**, returns the result array. This will return ``False`` if the input arrays have different sizes.
- ``LOGICAL_OR`` - takes two boolean arrays, does element wise **logical or**, returns the result array. This will return ``False`` if the input arrays have different sizes.

UNION, INTERSECT, EXCEPT
------------------------

Query can contain multiple :sql:`SELECT` statements, combined by one of the set operations - :sql:`UNION`, :sql:`INTERSECT` and :sql:`EXCEPT`.


Examples
~~~~~~~~

Querying for images containing 0 in `MNIST Train Dataset <https://app.activeloop.ai/activeloop/mnist-train>`_ with :meth:`ds.query <deeplake.core.dataset.Dataset.query>`.

>>> import deeplake
>>> ds = deeplake.load("hub://activeloop/mnist-train")
>>> result = ds.query("select * where labels == 0")
>>> len(result)
5923

Querying for samples with ``car`` or ``motorcycle`` in ``categories`` of `COCO Train Dataset <https://app.activeloop.ai/activeloop/coco-train>`_.

>>> import deeplake
>>> ds = deeplake.load("hub://activeloop/coco-train")
>>> result = ds.query("(select * where contains(categories, 'car')) union (select * where contains(categories, 'motorcycle'))")
>>> len(result)
14376
