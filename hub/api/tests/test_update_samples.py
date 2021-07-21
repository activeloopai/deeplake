import pytest
from typing import Callable
from hub.tests.common import assert_array_lists_equal
from hub.api.dataset import Dataset
import numpy as np


def _add_dummy_mnist(ds: Dataset):
    ds.create_tensor("images")
    ds.create_tensor("labels")

    ds.images.extend(np.ones((10, 28, 28)))
    ds.labels.extend(np.ones(10))

    return ds


def _make_update_assert_equal(ds_generator: Callable, tensor_name: str, index, value):
    """Updates a tensor and checks that the data is as expected.

    Example update:
        >>> ds.tensor[0:5] = [1, 2, 3, 4, 5]

    Args:
        ds_generator (Callable): Function that returns a new dataset object with each call.
        tensor_name (str): Name of the tensor to be updated.
        index (Any): Any value that can be used as an index for an access/modifier operation (`ds.tensor[index] = value`).
        value (Any): Any value that can be used as a value for an access/modifier operation (`ds.tensor[index] = value`).
    """

    ds = ds_generator()
    assert len(ds) == 10

    tensor = ds[tensor_name]
    expected = tensor.numpy()

    # make updates
    tensor[index] = value
    expected[index] = value

    # persistence
    ds = ds_generator()
    tensor = ds[tensor_name]

    assert_array_lists_equal(tensor.numpy(), expected)

    # make sure no new values are recorded
    ds = ds_generator()
    assert len(ds) == 10


def test_same_shape(local_ds_generator):
    gen = local_ds_generator

    _add_dummy_mnist(gen())

    # update single sample
    _make_update_assert_equal(gen, "images", 0, np.ones((28, 28)) * 5)
    _make_update_assert_equal(gen, "labels", 0, 5)
    _make_update_assert_equal(gen, "images", -1, np.ones((28, 28)) * 75)
    _make_update_assert_equal(gen, "labels", -5, 99)

    # update a range of samples
    _make_update_assert_equal(
        gen, "images", slice(0, 3), np.arange(3 * 28 * 28).reshape((3, 28, 28))
    )
    _make_update_assert_equal(gen, "labels", slice(0, 5), [1, 2, 3, 4, 5])


def test_new_shapes(local_ds_generator):
    gen = local_ds_generator
    _add_dummy_mnist(gen())

    # TODO: test updating a tensor that has already been indexed into. example:
    # t = ds.tensor[5:10]
    # t[0] = ...

    # TODO: test updating a tensor with an empty sample


def test_failures(memory_ds):
    _add_dummy_mnist(memory_ds)

    # primary axis doesn't match
    with pytest.raises(ValueError):
        memory_ds.images[0:3] = np.zeros((28, 28))
    with pytest.raises(ValueError):
        memory_ds.images[0:3] = np.zeros((2, 28, 28))
    with pytest.raises(ValueError):
        memory_ds.labels[0:3] = [1, 2, 3, 4]

    # dimensionality doesn't match
    with pytest.raises(ValueError):
        memory_ds.images[0:5] = np.zeros((5, 28))
    with pytest.raises(ValueError):
        memory_ds.labels[0:5] = np.zeros((5, 2))

    # inplace operators
    with pytest.raises(NotImplementedError):
        memory_ds.labels[0:5] += 1

    # make sure no data changed
    assert len(memory_ds.images) == 10
    assert len(memory_ds.labels) == 10
    np.tesing.assert_array_equal(memory_ds.images.numpy(), np.ones((10, 28, 28)))
    np.tesing.assert_array_equal(memory_ds.labels.numpy(), np.ones(10))
