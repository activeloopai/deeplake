Tensors
=======

.. currentmodule:: deeplake.core.dataset

Creating Tensors
~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.create_tensor
    Dataset.create_group
    Dataset.create_tensor_like


Deleting and Renaming Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    Dataset.delete_tensor
    Dataset.delete_group
    Dataset.rename_tensor
    Dataset.rename_group


.. currentmodule:: deeplake.core.tensor

Adding and deleting samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    Tensor.append
    Tensor.extend
    Tensor.pop
    Tensor.clear
    Tensor.__setitem__

Retrieving samples
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    Tensor.numpy
    Tensor.data
    Tensor.tobytes
    Tensor.text
    Tensor.dict
    Tensor.list
    Tensor._linked_sample

Tensor Properties
~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    Tensor.htype
    Tensor.base_htype
    Tensor.dtype
    Tensor.shape
    Tensor.shape_interval
    Tensor.ndim
    Tensor.num_samples
    Tensor.__len__
    Tensor.is_dynamic
    Tensor.is_sequence
    Tensor.is_link
    Tensor.verify

Info
~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    Tensor.info
    Tensor.sample_info

Video features
~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    Tensor.play
    Tensor.timestamps
