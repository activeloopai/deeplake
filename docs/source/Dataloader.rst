Dataloader
==========

Train your models using the high performance C++ dataloader, which is only available for registered users and datasets that are connected to Deep Lake.
This dataloader cannot be used with local datasets.

See the ``dataloader`` method on how to create dataloaders from your datasets:

.. currentmodule:: deeplake.core.dataset

.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.dataloader

DeepLakeDataLoader
~~~~~~~~~~~~~~~~~~

.. currentmodule:: deeplake.enterprise.dataloader

.. autoclass:: DeepLakeDataLoader()
    :members:
