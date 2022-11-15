.. currentmodule:: deeplake.core.dataset

Sampler
=======

The sampler applies weighted sampling on the dataset and returns the sampled view.
It creates a discrete distribution with given weights and randomly picks samples based on it.
The resulting view is generated in such a way that when creating a dataloader from the view and training on it,
the performance impact is minimal.
See the ``sample_by`` method on how to use this feature:

.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.sample_by