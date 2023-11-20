Vector Store
============
.. currentmodule:: deeplake.core.vectorstore.deeplake_vectorstore

Creating a Deep Lake Vector Store 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    VectorStore.__init__

Vector Store Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    VectorStore.add
    VectorStore.search
    VectorStore.delete
    VectorStore.delete_by_path
    VectorStore.update_embedding

Vector Store Properties
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    VectorStore.summary
    VectorStore.tensors
    VectorStore.__len__

VectorStore.DeepMemory
======================

Creating a Deep Memory
~~~~~~~~~~~~~~~~~~~~~~

if Deep Memory is available on your plan, it will be automatically initialized when you create a Vector Store.

.. currentmodule:: deeplake.core.vectorstore.deep_memory
.. autosummary::
    :toctree:
    :nosignatures:

    DeepMemory.__init__

Deep Memory Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    DeepMemory.train
    DeepMemory.cancel
    DeepMemory.delete

Deep Memory Properties
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    DeepMemory.status
    DeepMemory.list_jobs
    DeepMemory.__len__