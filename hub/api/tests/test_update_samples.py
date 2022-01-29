import pytest

from numpy import (
    ubyte, uint8,
    ones as np_ones,
    zeros as np_zeros,
    arange as np_arange,
    concatenate as np_concatenate,
    testing as np_testing,
)
from typing import Callable

from hub import read as hub_read
from hub.constants import KB
from hub.util.exceptions import TensorInvalidSampleShapeError
from hub.tests.common import assert_array_lists_equal


def _add_dummy_mnist(ds, **kwargs):
    compression = kwargs.get(
        "compression", {"image_compression": {"sample_compression": None}}
    )
    ds.create_tensor("images", htype="image", **compression["image_compression"])
    ds.create_tensor(
        "labels", htype="class_label", **compression.get("label_compression", {})
    )

    ds.images.extend(np_ones((10, 28, 28), dtype=uint8))
    ds.labels.extend(np_ones(10, dtype=uint8))

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


@pytest.mark.parametrize(
    "compression",
    [
        {
            "image_compression": {"sample_compression": None},
        },
        {
            "image_compression": {"sample_compression": None},
            "label_compression": {"sample_compression": "lz4"},
        },
        {
            "image_compression": {"sample_compression": None},
            "label_compression": {"chunk_compression": "lz4"},
        },
        {"image_compression": {"sample_compression": "png"}},
        {"image_compression": {"chunk_compression": "png"}},
        {"image_compression": {"sample_compression": "lz4"}},
        {"image_compression": {"chunk_compression": "lz4"}},
    ],
)
def test(local_ds_generator, compression):
    gen = local_ds_generator

    _add_dummy_mnist(gen(), **compression)

    # update single sample
    _make_update_assert_equal(
        gen, "images", -1, np_ones((1, 28, 28), dtype="uint8") * 75
    )  # same shape (with 1)
    _make_update_assert_equal(
        gen, "images", -1, np_ones((28, 28), dtype="uint8") * 75
    )  # same shape
    _make_update_assert_equal(
        gen, "images", 0, np_ones((28, 25), dtype="uint8") * 5
    )  # new shape
    _make_update_assert_equal(
        gen, "images", 0, np_ones((1, 32, 32), dtype="uint8") * 5
    )  # new shape (with 1)
    _make_update_assert_equal(
        gen, "images", -1, np_ones((0, 0), dtype="uint8")
    )  # empty sample (new shape)
    _make_update_assert_equal(gen, "labels", -5, uint8(99))
    _make_update_assert_equal(gen, "labels", 0, uint8(5))

    # update a range of samples
    x = np_arange(3 * 28 * 28).reshape((3, 28, 28)).astype("uint8")
    _make_update_assert_equal(gen, "images", slice(0, 3), x)  # same shapes
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np_zeros((2, 5, 28), dtype="uint8")
    )  # new shapes
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np_zeros((2, 5, 28), dtype=int).tolist()
    )  # test downcasting python scalars
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np_zeros((2, 5, 28), dtype=ubyte).tolist()
    )  # test upcasting
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np_zeros((2, 0, 0), dtype="uint8")
    )  # empty samples (new shape)
    _make_update_assert_equal(gen, "labels", slice(0, 5), [1, 2, 3, 4, 5])

    # update a range of samples with dynamic samples
    _make_update_assert_equal(
        gen,
        "images",
        slice(7, 10),
        [
            np_ones((28, 50), dtype="uint8") * 5,
            np_ones((0, 5), dtype="uint8"),
            np_ones((1, 1), dtype="uint8") * 10,
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
    ds.images.extend(np_zeros((10, 0, 0, 0), dtype=uint8))

    ds.images[0] = hub_read(cat_path)
    np_testing.assert_array_equal(ds.images[0].numpy(), hub_read(cat_path).array)

    ds.images[1] = [hub_read(flower_path)]
    np_testing.assert_array_equal(ds.images[1].numpy(), hub_read(flower_path).array)

    ds.images[8:10] = [hub_read(cat_path), hub_read(flower_path)]
    assert_array_lists_equal(
        ds.images[8:10].numpy(aslist=True),
        [hub_read(cat_path).array, hub_read(flower_path).array],
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

    np_testing.assert_array_equal([99, 98, 97], tensor[0])
    np_testing.assert_array_equal([44, 44, 44, 44], tensor[4])
    np_testing.assert_array_equal([33], tensor[5])

    assert tensor.shape_interval.lower == (6, 1)
    assert tensor.shape_interval.upper == (6, 5)
    assert len(tensor) == 6


def test_failures(memory_ds):
    _add_dummy_mnist(memory_ds)

    # primary axis doesn't match
    with pytest.raises(ValueError):
        memory_ds.images[0:3] = np_zeros((25, 30), dtype="uint8")
    with pytest.raises(ValueError):
        memory_ds.images[0:3] = np_zeros((2, 25, 30), dtype="uint8")
    with pytest.raises(TensorInvalidSampleShapeError):
        memory_ds.images[0] = np_zeros((2, 25, 30), dtype="uint8")
    with pytest.raises(ValueError):
        memory_ds.labels[0:3] = [1, 2, 3, 4]

    # dimensionality doesn't match
    with pytest.raises(TensorInvalidSampleShapeError):
        memory_ds.images[0:5] = np_zeros((5, 30), dtype="uint8")
    with pytest.raises(TensorInvalidSampleShapeError):
        memory_ds.labels[0:5] = np_zeros((5, 2, 3), dtype="uint8")

    # make sure no data changed
    assert len(memory_ds.images) == 10
    assert len(memory_ds.labels) == 10
    np_testing.assert_array_equal(
        memory_ds.images.numpy(), np_ones((10, 28, 28), dtype="uint8")
    )
    np_testing.assert_array_equal(
        memory_ds.labels.numpy(), np_ones((10, 1), dtype="uint8")
    )
    assert memory_ds.images.shape == (10, 28, 28)
    assert memory_ds.labels.shape == (10, 1)


def test_warnings(memory_ds):
    tensor = memory_ds.create_tensor("tensor", max_chunk_size=8 * KB)

    tensor.extend(np_ones((10, 12, 12), dtype="int32"))

    # this update makes (small) suboptimal chunks
    with pytest.warns(UserWarning):
        tensor[0:5] = np_zeros((5, 0, 0), dtype="int32")

    # this update makes (large) suboptimal chunks
    with pytest.warns(UserWarning):
        tensor[:] = np_zeros((10, 32, 31), dtype="int32")


@pytest.mark.parametrize(
    "compression",
    [
        {"sample_compression": None},
        {"sample_compression": "png"},
        {"chunk_compression": "png"},
        {"sample_compression": "lz4"},
        {"chunk_compression": "lz4"},
    ],
)
def test_inplace_updates(memory_ds, compression):
    ds = memory_ds
    ds.create_tensor("x", **compression)
    ds.x.extend(np_zeros((5, 32, 32, 3), dtype="uint8"))
    ds.x += 1
    np_testing.assert_array_equal(ds.x.numpy(), np_ones((5, 32, 32, 3)))
    ds.x += ds.x
    np_testing.assert_array_equal(ds.x.numpy(), np_ones((5, 32, 32, 3)) * 2)
    ds.x *= np_zeros(3, dtype="uint8")
    np_testing.assert_array_equal(ds.x.numpy(), np_zeros((5, 32, 32, 3)))
    ds.x += 6
    ds.x //= 2
    np_testing.assert_array_equal(ds.x.numpy(), np_ones((5, 32, 32, 3)) * 3)
    ds.x[:3] *= 0
    np_testing.assert_array_equal(
        ds.x.numpy(),
        np_concatenate([np_zeros((3, 32, 32, 3)), np_ones((2, 32, 32, 3)) * 3]),
    )

    # Different shape
    ds.x.append(np_zeros((100, 50, 3), dtype="uint8"))
    ds.x[5] += 1
    np_testing.assert_array_equal(ds.x[5].numpy(), np_ones((100, 50, 3)))
    np_testing.assert_array_equal(
        ds.x[:5].numpy(),
        np_concatenate([np_zeros((3, 32, 32, 3)), np_ones((2, 32, 32, 3)) * 3]),
    )
    ds.x[:5] *= 0
    np_testing.assert_array_equal(ds.x[:5].numpy(), np_zeros((5, 32, 32, 3)))
    np_testing.assert_array_equal(ds.x[5].numpy(), np_ones((100, 50, 3)))
