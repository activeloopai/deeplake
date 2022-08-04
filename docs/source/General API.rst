General API
===========
.. automodule:: hub
.. currentmodule:: hub

Dataset Creation
~~~~~~~~~~~~~~~~
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
.. autofunction:: load

Deleting Datasets
~~~~~~~~~~~~~~~~~
.. autofunction:: delete

Renaming Datasets
~~~~~~~~~~~~~~~~~
.. autofunction:: rename

Helper Functions
~~~~~~~~~~~~~~~~
.. autofunction:: list

.. autofunction:: exists

Htypes
~~~~~~

.. automodule:: hub.htype

Reading data
~~~~~~~~~~~~

.. autofunction:: hub.read

Linked data
~~~~~~~~~~~

.. autofunction:: hub.link

Tiling
~~~~~~

.. autofunction:: hub.tiled

Parallelism
~~~~~~~~~~~

.. autofunction:: hub.compute
.. autofunction:: hub.compose