from hub.constants import KB
from hub.util.exceptions import (
    InvalidSubsliceUpdateShapeError,
    TensorInvalidSampleShapeError,
)
import pytest
from hub.tests.common import assert_array_lists_equal
import numpy as np
import hub
from hub.tests.common import compressions


def _add_dummy_mnist(ds, **kwargs):
    compression = kwargs.get(
        "compression", {"image_compression": {"sample_compression": None}}
    )
    ds.create_tensor("images", htype="image", **compression["image_compression"])
    ds.create_tensor(
        "labels", htype="class_label", **compression.get("label_compression", {})
    )

    ds.images.extend(np.ones((10, 28, 28), dtype=np.uint8))
    ds.labels.extend(np.ones(10, dtype=np.uint8))

    return ds


@compressions
def test(local_ds_generator, compression):
    ds = local_ds_generator()
    ds.create_tensor("images", **compression)
    ds.images.extend(np.zeros((10, 28, 28), dtype=np.uint8))

    ds = local_ds_generator()
    ds.images[0] = np.ones((25, 30), dtype=np.uint8)
    ds.images[1] = np.ones((3, 4), dtype=np.uint8)

    ds.images[2:5] = np.ones((3, 5, 5), dtype=np.uint8)
    ds.images[6:8, 10:20, 0] = np.ones((2, 10), dtype=np.uint8) * 5

    ds = local_ds_generator()
    ds.images[9] = np.ones((0, 10), dtype=np.uint8)

    ds = local_ds_generator()

    expected = list(np.zeros((10, 28, 28), dtype="uint8"))
    expected[0] = np.ones((25, 30), dtype="uint8")
    expected[1] = np.ones((3, 4), dtype="uint8")
    expected[2:5] = np.ones((3, 5, 5), dtype="uint8")

    expected[6][10:20, 0] = np.ones((10), dtype="uint8") * 5
    expected[7][10:20, 0] = np.ones((10), dtype="uint8") * 5

    expected[9] = np.ones((0, 10), dtype="uint8")

    assert_array_lists_equal(ds.images.numpy(aslist=True), expected)

    assert ds.images.shape_interval.lower == (10, 0, 4)
    assert ds.images.shape_interval.upper == (10, 28, 30)
    assert ds.images.num_chunks == 1


@compressions
def test_subslice(local_ds_generator, compression):
    ds = local_ds_generator()

    expected_0 = np.ones((10, 10, 3), dtype="uint8")
    expected_0[1:5, -5:-1, 1] = np.zeros((4, 4), dtype="uint8")

    ds.create_tensor("image", htype="image", **compression)
    ds.image.extend(np.ones((10, 10, 10, 3), dtype="uint8"))

    # TODO: implement and uncomment check when negative indexing is implemented
    with pytest.raises(NotImplementedError):
        ds.image[0, 1:5, -5:-1, 1] = np.zeros((4, 4))
    assert ds.image.num_chunks == 1
        
    # np.testing.assert_array_equal(
    #     ds.image[1:].numpy(), np.ones((9, 10, 10, 3), dtype="uint8")
    # )
    # np.testing.assert_array_equal(ds.image[0].numpy(), expected_0)


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
    assert ds.images.num_chunks == 1


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


def test_subslice_failure(memory_ds):
    memory_ds.create_tensor("tensor")
    memory_ds.tensor.extend(np.ones((3, 28, 28, 3)))
    memory_ds.tensor.append(np.ones((28, 35, 3)))

    # when updating a sample's subslice, the shape MUST match the subslicing.
    # this is different than updating samples entirely, where the new sample
    # may have a larger/smaller shape.
    with pytest.raises(InvalidSubsliceUpdateShapeError):
        memory_ds.tensor[1, 10:20, 5:10, :] = np.zeros((2, 10, 5, 3))
    with pytest.raises(InvalidSubsliceUpdateShapeError):
        memory_ds.tensor[1, 10:20, 5:10, :] = np.zeros((0, 10, 5, 3))
    with pytest.raises(InvalidSubsliceUpdateShapeError):
        memory_ds.tensor[1, 10:20, 5:10, :] = np.zeros((1, 20, 5, 3))
    with pytest.raises(InvalidSubsliceUpdateShapeError):
        memory_ds.tensor[1, 10:20, 5:10, :] = np.zeros((1, 9, 5, 3))
    with pytest.raises(InvalidSubsliceUpdateShapeError):
        memory_ds.tensor[1, 10:20, 5:10, :] = np.zeros((1, 10, 6, 3))
    with pytest.raises(InvalidSubsliceUpdateShapeError):
        memory_ds.tensor[1, 10:20, 5:10, :] = np.zeros((1, 10, 4, 3))
    with pytest.raises(InvalidSubsliceUpdateShapeError):
        memory_ds.tensor[1, 10:20, 5:10, :] = np.zeros((1, 10, 5, 1))
    with pytest.raises(InvalidSubsliceUpdateShapeError):
        memory_ds.tensor[1, 10:20, 5:10, :] = np.zeros((1, 10, 5, 4))
    with pytest.raises(InvalidSubsliceUpdateShapeError):
        memory_ds.tensor[1, 10:20, 5:10, :] = np.zeros((1, 10, 5, 4))

    assert memory_ds.tensor.shape_interval.lower == (4, 28, 28, 3)
    assert memory_ds.tensor.shape_interval.upper == (4, 28, 35, 3)

    np.testing.assert_array_equal(memory_ds.tensor[:3].numpy(), np.ones((3, 28, 28, 3)))
    np.testing.assert_array_equal(memory_ds.tensor[3].numpy(), np.ones((28, 35, 3)))


def test_small_warning(memory_ds):
    tensor = memory_ds.create_tensor("tensor", max_chunk_size=8 * KB)

    tensor.extend(np.ones((10, 12, 12), dtype="int32"))

    # this update makes (small) suboptimal chunks
    with pytest.warns(UserWarning):
        tensor[0:5] = np.zeros((5, 0, 0), dtype="int32")


def test_large_warning(memory_ds):
    tensor = memory_ds.create_tensor("tensor", max_chunk_size=8 * KB)

    tensor.extend(np.ones((10, 12, 12), dtype="int32"))

    # this update makes (large) suboptimal chunks
    with pytest.warns(UserWarning):
        tensor[:] = np.zeros((10, 32, 31), dtype="int32")


@pytest.mark.parametrize(
    "compression",
    [
        {"sample_compression": None},
        {"sample_compression": "png"},
        {"chunk_compression": "png"},
    ],
)
def test_inplace_updates(memory_ds, compression):
    ds = memory_ds
    ds.create_tensor("x", **compression)
    ds.x.extend(np.zeros((5, 32, 32, 3), dtype="uint8"))
    ds.x += 1
    np.testing.assert_array_equal(ds.x.numpy(), np.ones((5, 32, 32, 3)))
    ds.x += ds.x
    np.testing.assert_array_equal(ds.x.numpy(), np.ones((5, 32, 32, 3)) * 2)
    ds.x *= np.zeros(3, dtype="uint8")
    np.testing.assert_array_equal(ds.x.numpy(), np.zeros((5, 32, 32, 3)))
    ds.x += 6
    ds.x //= 2
    np.testing.assert_array_equal(ds.x.numpy(), np.ones((5, 32, 32, 3)) * 3)
    ds.x[:3] *= 0
    np.testing.assert_array_equal(
        ds.x.numpy(),
        np.concatenate([np.zeros((3, 32, 32, 3)), np.ones((2, 32, 32, 3)) * 3]),
    )

    # Different shape
    ds.x.append(np.zeros((100, 50, 3), dtype="uint8"))
    ds.x[5] += 1
    np.testing.assert_array_equal(ds.x[5].numpy(), np.ones((100, 50, 3)))
    np.testing.assert_array_equal(
        ds.x[:5].numpy(),
        np.concatenate([np.zeros((3, 32, 32, 3)), np.ones((2, 32, 32, 3)) * 3]),
    )
    ds.x[:5] *= 0
    np.testing.assert_array_equal(ds.x[:5].numpy(), np.zeros((5, 32, 32, 3)))
    np.testing.assert_array_equal(ds.x[5].numpy(), np.ones((100, 50, 3)))

    assert ds.x.num_chunks == 1
