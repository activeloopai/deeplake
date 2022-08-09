Datasets
========
.. currentmodule:: hub

Creating Datasets
~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    hub.dataset
    hub.empty
    hub.like
    hub.ingest
    hub.ingest_kaggle
    hub.ingest_dataframe
    hub.ingest_huggingface

Loading Datasets
~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    hub.load

Deleting and Renaming Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    hub.delete
    hub.rename

Copying Datasets
~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:
    
    hub.copy
    hub.deepcopy

.. currentmodule:: hub.core.dataset

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
    Dataset.read_only
    Dataset.token
    Dataset.info

Dataset Views
~~~~~~~~~~~~~
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
    :toctree:
    :nosignatures:

    Dataset.filter
    Dataset.save_view
    Dataset.get_view
    Dataset.load_view
    Dataset.delete_view
    Dataset.get_views
    Dataset.is_view

Version Control
~~~~~~~~~~~~~~~
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