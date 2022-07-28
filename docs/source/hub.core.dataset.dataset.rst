hub.core.dataset.dataset
========================
.. currentmodule:: hub.core.dataset.dataset

Tensor Operations
~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    create_tensor
    create_group
    create_tensor_like
    delete_tensor
    delete_group
    rename_tensor
    rename_group

Dataset Operations
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    summary
    append
    extend
    copy
    delete
    rename
    visualize
    pop
    rechunk
    flush
    clear_cache
    size_approx
    
Dataset Credentials
~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    add_creds_key
    populate_creds
    update_creds_key
    change_creds_management
    get_creds_keys

Dataset Properties
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensors
    groups
    read_only
    token
    info

Dataset Views
~~~~~~~~~~~~~
Views of a dataset can be created through indexing or filtering a :class:`Dataset`.
A dataset can be a view of an existing dataset. Views point to the samples of the existing dataset.


Version Control
~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    commit
    diff
    checkout
    merge
    log
    reset
    get_commit_details
    commit_id
    branch
    pending_commit_id
    has_head_changes
    commits
    branches

.. autoclass:: Dataset()
    :members: