Datasets
========
.. currentmodule:: deeplake

.. _creating-datasets:

Creating Datasets
~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    deeplake.dataset
    deeplake.empty
    deeplake.like
    deeplake.ingest
    deeplake.ingest_kaggle
    deeplake.ingest_dataframe
    deeplake.ingest_huggingface

Loading Datasets
~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    deeplake.load

Deleting and Renaming Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    deeplake.delete
    deeplake.rename

Copying Datasets
~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:
    
    deeplake.copy
    deeplake.deepcopy

.. currentmodule:: deeplake.core.dataset

Dataset Operations
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.summary
    Dataset.append
    Dataset.extend
    Dataset.copy
    Dataset.delete
    Dataset.rename
    Dataset.visualize
    Dataset.pop
    Dataset.rechunk
    Dataset.flush
    Dataset.clear_cache
    Dataset.size_approx

Dataset Visualization
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.visualize
    
Dataset Credentials
~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.add_creds_key
    Dataset.populate_creds
    Dataset.update_creds_key
    Dataset.change_creds_management
    Dataset.get_creds_keys

Dataset Properties
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.tensors
    Dataset.groups
    Dataset.num_samples
    Dataset.read_only
    Dataset.info

Dataset Version Control
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.commit
    Dataset.diff
    Dataset.checkout
    Dataset.merge
    Dataset.log
    Dataset.reset
    Dataset.get_commit_details
    Dataset.commit_id
    Dataset.branch
    Dataset.pending_commit_id
    Dataset.has_head_changes
    Dataset.commits
    Dataset.branches

Dataset Views
~~~~~~~~~~~~~
A dataset view is a subset of a dataset that points to specific samples (indices) in an existing dataset. Dataset views
can be created using indexing or querying a dataset using :func:`~Dataset.filter`. Dataset views can only be saved when a dataset
has been committed and has no changes on the HEAD node, in order to preserve data lineage and prevent the underlying data from
changing after the query or filter conditions have been evaluated.

    >>> import deeplake
    >>> # load dataset
    >>> ds = deeplake.load("deeplake://activeloop/mnist-train")
    >>> # filter dataset
    >>> zeros = ds.filter("labels == 0")
    >>> # save view
    >>> zeros.save_view(id="zeros")
    >>> # load_view
    >>> zeros = ds.load_view(id="zeros")
    >>> len(zeros)
    5923

.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.filter
    Dataset.save_view
    Dataset.get_view
    Dataset.load_view
    Dataset.delete_view
    Dataset.get_views
    Dataset.is_view
