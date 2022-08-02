hub.core.tensor
===============
.. currentmodule:: hub.core.tensor

Tensor
~~~~~~

Data is stored in tensors within datasets. They are created using Dataset's :meth:`Dataset.create_tensor <hub.core.dataset.Dataset.create_tensor>` method 
and can be accessed by ``__getitem__`` or ``__getattr__`` of datasets. Tensors can be indexed just like numpy arrays. 
No data is fetched until you call one of the functions in :ref:`Retrieving samples`.

>>> ds = hub.empty("./my_dataset")
>>> ds.create_tensor('abc')
>>> ds.abc.append([1, 2, 3, 4])
>>> ds.abc # tensor abc
>>> ds['abc'] # tensor abc
>>> ds.abc.numpy()
array([[1, 2, 3, 4]])
>>> ds['abc'].numpy()
array([[1, 2, 3, 4]])


Adding and deleting samples
---------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Tensor.append
    ~Tensor.extend
    ~Tensor.pop
    ~Tensor.clear
    ~Tensor.__setitem__

Retrieving samples
------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Tensor.numpy
    ~Tensor.data
    ~Tensor.tobytes
    ~Tensor.text
    ~Tensor.dict
    ~Tensor.list

Tensor Properties
-----------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Tensor.htype
    ~Tensor.base_htype
    ~Tensor.dtype
    ~Tensor.shape
    ~Tensor.shape_interval
    ~Tensor.ndim
    ~Tensor.num_samples
    ~Tensor.is_dynamic
    ~Tensor.is_sequence
    ~Tensor.is_link
    ~Tensor.verify
    ~Tensor.meta

Info
----

.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Tensor.info
    ~Tensor.sample_info

Video features
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    ~Tensor.play
    ~Tensor.timestamps
