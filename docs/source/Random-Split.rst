.. currentmodule:: deeplake.core.dataset

Random Split
============

Splits the dataset into non overlapping new datasets of given lengths.
The resulting datasets are generated in such a way that when creating a dataloader from the view and training on it,
the performance impact is minimal. Using the outputs of this function with .pytorch method of dataset (instead of .dataloader) may result in poor performance.
See the ``random_split`` method on how to use this feature:

.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.random_split