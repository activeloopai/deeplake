import pytest
from typing import Callable
from hub.tests.common import assert_array_lists_equal
from hub.api.dataset import Dataset
import numpy as np


def _add_dummy_mnist(ds: Dataset, images_compression: str = None):
    ds.create_tensor("images", sample_compression=images_compression)
    ds.create_tensor("labels")

    ds.images.extend(np.ones((10, 28, 28)))
    ds.labels.extend(np.ones(10))

    return ds


def _make_update_assert_equal(
    ds_generator: Callable, tensor_name: str, index, value, pre_index=None
):
    """Updates a tensor and checks that the data is as expected.

    Example update:
        >>> ds.tensor[0:5] = [1, 2, 3, 4, 5]

    Args:
        ds_generator (Callable): Function that returns a new dataset object with each call.
        tensor_name (str): Name of the tensor to be updated.
        index (Any): Any value that can be used as an index for updating (`ds.tensor[index] = value`).
        value (Any): Any value that can be used as a value for updating (`ds.tensor[index] = value`).
        pre_index (Any): Any value that can be used as an index. This simulates indexing a tensor and then
            indexing again to update. (`ds.tensor[pre_index][index] = value`).
    """

    ds = ds_generator()
    assert len(ds) == 10

    tensor = ds[tensor_name]
    expected = tensor.numpy(aslist=True)

    # this is necessary because `expected` uses `aslist=True` to handle dynamic cases.
    # with `aslist=False`, this wouldn't be necessary.
    expected_value = value
    if hasattr(value, "__len__"):
        if len(value) == 1:
            expected_value = value[0]

    # make updates
    if pre_index is None:
        tensor[index] = value
        expected[index] = expected_value
    else:
        tensor[pre_index][index] = value
        expected[pre_index][index] = expected_value

    # non-persistence check
    actual = tensor.numpy(aslist=True)
    assert_array_lists_equal(actual, expected)

    # persistence check
    ds = ds_generator()
    tensor = ds[tensor_name]
    actual = tensor.numpy(aslist=True)
    assert_array_lists_equal(actual, expected)

    # make sure no new values are recorded
    ds = ds_generator()
    assert len(ds) == 10


@pytest.mark.parametrize("images_compression", [None, "png"])
def test(local_ds_generator, images_compression):
    gen = local_ds_generator

    _add_dummy_mnist(gen(), images_compression=images_compression)

    # update single sample
    _make_update_assert_equal(
        gen, "images", -1, np.ones((1, 28, 28)) * 75
    )  # same shape (with 1)
    _make_update_assert_equal(gen, "images", -1, np.ones((28, 28)) * 75)  # same shape
    _make_update_assert_equal(gen, "images", 0, np.ones((28, 25)) * 5)  # new shape
    _make_update_assert_equal(
        gen, "images", 0, np.ones((1, 28, 25)) * 5
    )  # new shape (with 1)
    _make_update_assert_equal(
        gen, "images", -1, np.ones((0, 0))
    )  # empty sample (new shape)
    _make_update_assert_equal(gen, "labels", -5, 99)
    _make_update_assert_equal(gen, "labels", 0, 5)

    # update a range of samples
    x = np.arange(3 * 28 * 28).reshape((3, 28, 28))
    _make_update_assert_equal(gen, "images", slice(0, 3), x)  # same shapes
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np.zeros((2, 5, 28))
    )  # new shapes
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np.zeros((2, 0, 0))
    )  # empty samples (new shape)
    _make_update_assert_equal(gen, "labels", slice(0, 5), [1, 2, 3, 4, 5])

    # update a range of samples with dynamic samples
    _make_update_assert_equal(
        gen,
        "images",
        slice(7, 10),
        [
            np.ones((28, 50)) * 5,
            np.ones((0, 5)),
            np.ones((1, 1)) * 10,
        ],
    )

    # TODO: hub.read test


def test_pre_indexed_tensor(local_ds_generator):
    """A pre-indexed tensor update means the tensor was already indexed into, and an update is being made to that tensor view.

    Example:
        >>> tensor = ds.tensor[0:10]
        >>> len(tensor)
        10
        >>> tensor[0:5] = ...
    """

    gen = local_ds_generator
    _add_dummy_mnist(gen())

    x = np.arange(3 * 28 * 28).reshape((3, 28, 28))
    _make_update_assert_equal(gen, "images", slice(4, 7), x, pre_index=slice(2, 9))

    _make_update_assert_equal(
        gen, "labels", slice(0, 5), [1, 2, 3, 4, 5], pre_index=slice(0, 6)
    )


def test_failures(memory_ds):
    _add_dummy_mnist(memory_ds)

    # primary axis doesn't match
    with pytest.raises(ValueError):
        memory_ds.images[0:3] = np.zeros((28, 28))
    with pytest.raises(ValueError):
        memory_ds.images[0:3] = np.zeros((2, 28, 28))
    with pytest.raises(ValueError):
        memory_ds.images[0] = np.zeros((2, 28, 28))
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
