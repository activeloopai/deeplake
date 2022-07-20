Hub
===
.. automodule:: hub
.. currentmodule:: hub

Htype
=====
.. automodule:: hub.htype

Dataset
=======
.. currentmodule:: hub.core.dataset.dataset
.. autoclass:: Dataset()
    :members:

Creation
~~~~~~~~
.. currentmodule:: hub
.. autosummary::
    :toctree: generated
    :nosignatures:

    dataset
    empty
    list
    ingest
    ingest_kaggle
    ingest_dataframe
    ingest_huggingface

Copying
~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    copy
    deepcopy

Loading
~~~~~~~
.. autofunction:: load

Deletion
~~~~~~~~
.. autofunction:: delete

Renaming
~~~~~~~~
.. autofunction:: rename

List
~~~~
.. autofunction:: list


Tensor
======
.. currentmodule:: hub.core.tensor
.. autoclass:: Tensor()
    :members:

Creation
~~~~~~~~
See :func: 
.. currentmodule:: hub.core.dataset.dataset
.. autosummary::
    :toctree: generated
    :nosignatures:

    Dataset.create_tensor
    Dataset.create_tensor_like