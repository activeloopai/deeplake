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
    deeplake.ingest_classification
    deeplake.ingest_coco
    deeplake.ingest_yolo
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
    Dataset.query
    Dataset.copy
    Dataset.delete
    Dataset.rename
    Dataset.connect
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
    Dataset.max_len
    Dataset.min_len

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
can be created by indexing a dataset, filtering a dataset with :meth:`Dataset.filter`, querying a dataset with :meth:`Dataset.query`
or by sampling a dataset with :meth:`Dataset.sample_by`.
Filtering is done with user-defined functions or simplified expressions whereas query can perform SQL-like queries with our 
Tensor Query Language. See the full TQL spec :ref:`here <tql>`.


Dataset views can only be saved when a dataset has been committed and has no changes on the HEAD node, 
in order to preserve data lineage and prevent the underlying data from changing after the query or filter conditions have been evaluated.

**Example**

>>> import deeplake
>>> # load dataset
>>> ds = deeplake.load("hub://activeloop/mnist-train")
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

    Dataset.query
    Dataset.sample_by
    Dataset.filter
    Dataset.save_view
    Dataset.get_view
    Dataset.load_view
    Dataset.delete_view
    Dataset.get_views
    Dataset.is_view
    Dataset.min_view
    Dataset.max_view
