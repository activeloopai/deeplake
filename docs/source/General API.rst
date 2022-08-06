General API
===========
.. automodule:: hub
.. currentmodule:: hub

Creating Datasets
~~~~~~~~~~~~~~~~~
.. currentmodule:: hub
.. autosummary::
    :toctree: generated
    :nosignatures:

    dataset
    empty
    like
    ingest
    ingest_kaggle
    ingest_dataframe
    ingest_huggingface

Copying Datasets
~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    copy
    deepcopy

Loading Datasets
~~~~~~~~~~~~~~~~

hub.load
--------

.. autofunction:: load

Deleting Datasets
~~~~~~~~~~~~~~~~~

hub.delete
----------

.. autofunction:: delete

Renaming Datasets
~~~~~~~~~~~~~~~~~

hub.rename
----------

.. autofunction:: rename

Helper Functions
~~~~~~~~~~~~~~~~

hub.list
--------
.. autofunction:: list

hub.exists
----------
.. autofunction:: exists

Htypes
~~~~~~

.. automodule:: hub.htype

Reading data
~~~~~~~~~~~~

hub.read
--------
.. autofunction:: hub.read

Linked data
~~~~~~~~~~~

hub.link
--------
.. autofunction:: hub.link

Tiling
~~~~~~

hub.tiled
---------
.. autofunction:: hub.tiled

Parallelism
~~~~~~~~~~~

hub.compute
-----------
.. autofunction:: hub.compute

hub.compose
-----------
.. autofunction:: hub.compose