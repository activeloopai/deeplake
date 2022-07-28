hub.core.dataset
================
.. currentmodule:: hub.core.dataset

Dataset
~~~~~~~

.. autoclass:: Dataset()

Tensor Operations
-----------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Dataset.create_tensor
    ~Dataset.create_group
    ~Dataset.create_tensor_like
    ~Dataset.delete_tensor
    ~Dataset.delete_group
    ~Dataset.rename_tensor
    ~Dataset.rename_group

Dataset Operations
------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Dataset.summary
    ~Dataset.append
    ~Dataset.extend
    ~Dataset.copy
    ~Dataset.delete
    ~Dataset.rename
    ~Dataset.visualize
    ~Dataset.pop
    ~Dataset.rechunk
    ~Dataset.flush
    ~Dataset.clear_cache
    ~Dataset.size_approx
    
Dataset Credentials
-------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Dataset.add_creds_key
    ~Dataset.populate_creds
    ~Dataset.update_creds_key
    ~Dataset.change_creds_management
    ~Dataset.get_creds_keys

Dataset Properties
------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Dataset.tensors
    ~Dataset.groups
    ~Dataset.read_only
    ~Dataset.token
    ~Dataset.info

Dataset Views
-------------
A dataset can be a view of an existing dataset. Views point to the samples of the existing dataset in a specific commit.
They can be created and saved by indexing or querying a dataset through :func:`~Dataset.filter`.

    >>> import hub
    >>> # load dataset
    >>> ds = hub.load("hub://activeloop/mnist-train")
    >>> # filter dataset
    >>> zeros = ds.filter("labels == 0")
    >>> # save view
    >>> zeros.save_view(id="zeros")
    >>> # load_view
    >>> zeros = ds.load_view(id="zeros")
    >>> len(zeros)
    5923

.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Dataset.filter
    ~Dataset.save_view
    ~Dataset.get_view
    ~Dataset.load_view
    ~Dataset.delete_view
    ~Dataset.get_views
    ~Dataset.is_view

Version Control
---------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Dataset.commit
    ~Dataset.diff
    ~Dataset.checkout
    ~Dataset.merge
    ~Dataset.log
    ~Dataset.reset
    ~Dataset.get_commit_details
    ~Dataset.commit_id
    ~Dataset.branch
    ~Dataset.pending_commit_id
    ~Dataset.has_head_changes
    ~Dataset.commits
    ~Dataset.branches

PyTorch and Tensorflow support
------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Dataset.pytorch
    ~Dataset.tensorflow

HubCloudDataset
~~~~~~~~~~~~~~~
.. autoclass:: HubCloudDataset()
    :members:

ViewEntry
~~~~~~~~~
.. autoclass:: ViewEntry()
    :members: