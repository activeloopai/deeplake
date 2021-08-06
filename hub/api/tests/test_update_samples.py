from hub.constants import KB
from hub.util.exceptions import TensorInvalidSampleShapeError
import pytest
from typing import Callable
from hub.tests.common import assert_array_lists_equal, update_chunk_sizes
import numpy as np
import hub


def _add_dummy_mnist(ds, images_compression: str = None):
    ds.create_tensor("images", htype="image", sample_compression=images_compression)
    ds.create_tensor("labels", htype="class_label")

    ds.images.extend(np.ones((10, 28, 28), dtype=np.uint8))
    ds.labels.extend(np.ones(10, dtype=np.uint8))

    return ds


def _make_update_assert_equal(
    ds_generator: Callable,
    tensor_name: str,
    index,
    value,
    check_persistence: bool = True,
):
    """Updates a tensor and checks that the data is as expected.

    Example update:
        >>> ds.tensor[0:5] = [1, 2, 3, 4, 5]

    Args:
        ds_generator (Callable): Function that returns a new dataset object with each call.
        tensor_name (str): Name of the tensor to be updated.
        index (Any): Any value that can be used as an index for updating (`ds.tensor[index] = value`).
        value (Any): Any value that can be used as a value for updating (`ds.tensor[index] = value`).
        check_persistence (bool): If True, the update will be tested to make sure it can be serialized/deserialized.
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
    tensor[index] = value
    expected[index] = expected_value

    # non-persistence check
    actual = tensor.numpy(aslist=True)
    assert_array_lists_equal(actual, expected)
    assert len(ds) == 10

    if check_persistence:
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
        gen, "images", -1, np.ones((1, 28, 28), dtype="uint8") * 75
    )  # same shape (with 1)
    _make_update_assert_equal(
        gen, "images", -1, np.ones((28, 28), dtype="uint8") * 75
    )  # same shape
    _make_update_assert_equal(
        gen, "images", 0, np.ones((28, 25), dtype="uint8") * 5
    )  # new shape
    _make_update_assert_equal(
        gen, "images", 0, np.ones((1, 32, 32), dtype="uint8") * 5
    )  # new shape (with 1)
    _make_update_assert_equal(
        gen, "images", -1, np.ones((0, 0), dtype="uint8")
    )  # empty sample (new shape)
    _make_update_assert_equal(gen, "labels", -5, np.uint8(99))
    _make_update_assert_equal(gen, "labels", 0, np.uint8(5))

    # update a range of samples
    x = np.arange(3 * 28 * 28).reshape((3, 28, 28)).astype("uint8")
    _make_update_assert_equal(gen, "images", slice(0, 3), x)  # same shapes
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np.zeros((2, 5, 28), dtype="uint8")
    )  # new shapes
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np.zeros((2, 5, 28), dtype=int).tolist()
    )  # test downcasting python scalars
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np.zeros((2, 5, 28), dtype=np.ubyte).tolist()
    )  # test upcasting
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np.zeros((2, 0, 0), dtype="uint8")
    )  # empty samples (new shape)
    _make_update_assert_equal(gen, "labels", slice(0, 5), [1, 2, 3, 4, 5])

    # update a range of samples with dynamic samples
    _make_update_assert_equal(
        gen,
        "images",
        slice(7, 10),
        [
            np.ones((28, 50), dtype="uint8") * 5,
            np.ones((0, 5), dtype="uint8"),
            np.ones((1, 1), dtype="uint8") * 10,
        ],
    )

    ds = gen()
    assert ds.images.shape_interval.lower == (10, 0, 0)
    assert ds.images.shape_interval.upper == (10, 32, 50)


@pytest.mark.parametrize("images_compression", [None, "png"])
def test_hub_read(local_ds_generator, images_compression, cat_path, flower_path):
    gen = local_ds_generator

    ds = gen()
    ds.create_tensor("images", htype="image", sample_compression=images_compression)
    ds.images.extend(np.zeros((10, 0, 0, 0), dtype=np.uint8))

    ds.images[0] = hub.read(cat_path)
    np.testing.assert_array_equal(ds.images[0].numpy(), hub.read(cat_path).array)

    ds.images[1] = [hub.read(flower_path)]
    np.testing.assert_array_equal(ds.images[1].numpy(), hub.read(flower_path).array)

    ds.images[8:10] = [hub.read(cat_path), hub.read(flower_path)]
    assert_array_lists_equal(
        ds.images[8:10].numpy(aslist=True),
        [hub.read(cat_path).array, hub.read(flower_path).array],
    )

    assert ds.images.shape_interval.lower == (10, 0, 0, 0)
    assert ds.images.shape_interval.upper == (10, 900, 900, 4)

    assert len(ds.images) == 10


def test_pre_indexed_tensor(memory_ds):
    """A pre-indexed tensor update means the tensor was already indexed into, and an update is being made to that tensor view."""

    tensor = memory_ds.create_tensor("tensor")

    tensor.append([0, 1, 2])
    tensor.append([3, 4, 5, 6, 7])
    tensor.append([8, 5])
    tensor.append([9, 10, 11])
    tensor.append([12, 13, 14, 15, 16])
    tensor.append([17, 18, 19, 20, 21])

    tensor[0:5][0] = [99, 98, 97]
    tensor[5:10][0] = [44, 44, 44, 44]
    tensor[4:10][0:2] = [[44, 44, 44, 44], [33]]

    np.testing.assert_array_equal([99, 98, 97], tensor[0])
    np.testing.assert_array_equal([44, 44, 44, 44], tensor[4])
    np.testing.assert_array_equal([33], tensor[5])

    assert tensor.shape_interval.lower == (6, 1)
    assert tensor.shape_interval.upper == (6, 5)
    assert len(tensor) == 6


def test_failures(memory_ds):
    _add_dummy_mnist(memory_ds)

    # primary axis doesn't match
    with pytest.raises(ValueError):
        memory_ds.images[0:3] = np.zeros((25, 30), dtype="uint8")
    with pytest.raises(ValueError):
        memory_ds.images[0:3] = np.zeros((2, 25, 30), dtype="uint8")
    with pytest.raises(TensorInvalidSampleShapeError):
        memory_ds.images[0] = np.zeros((2, 25, 30), dtype="uint8")
    with pytest.raises(ValueError):
        memory_ds.labels[0:3] = [1, 2, 3, 4]

    # dimensionality doesn't match
    with pytest.raises(TensorInvalidSampleShapeError):
        memory_ds.images[0:5] = np.zeros((5, 30), dtype="uint8")
    with pytest.raises(TensorInvalidSampleShapeError):
        memory_ds.labels[0:5] = np.zeros((5, 2, 3), dtype="uint8")

    # inplace operators
    with pytest.raises(NotImplementedError):
        memory_ds.labels[0:5] += 1

    # make sure no data changed
    assert len(memory_ds.images) == 10
    assert len(memory_ds.labels) == 10
    np.testing.assert_array_equal(
        memory_ds.images.numpy(), np.ones((10, 28, 28), dtype="uint8")
    )
    np.testing.assert_array_equal(
        memory_ds.labels.numpy(), np.ones((10, 1), dtype="uint8")
    )
    assert memory_ds.images.shape == (10, 28, 28)
    assert memory_ds.labels.shape == (10, 1)


def test_warnings(memory_ds):
    tensor = memory_ds.create_tensor("tensor")

    # this MUST be after all tensors have been created!
    update_chunk_sizes(memory_ds, 8 * KB)

    tensor.extend(np.ones((10, 12, 12), dtype="int32"))

    # this update makes (small) suboptimal chunks
    with pytest.warns(UserWarning):
        tensor[0:5] = np.zeros((5, 0, 0), dtype="int32")

    # this update makes (large) suboptimal chunks
    with pytest.warns(UserWarning):
        tensor[:] = np.zeros((10, 32, 31), dtype="int32")
