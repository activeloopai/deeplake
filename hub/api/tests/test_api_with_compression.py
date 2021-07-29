from hub.util.exceptions import (
    SampleCompressionError,
    TensorMetaMissingRequiredValue,
    UnsupportedCompressionError,
)
import pytest
from hub.core.tensor import Tensor
from hub.tests.common import TENSOR_KEY
from hub.tests.dataset_fixtures import enabled_datasets
import numpy as np

import hub
from hub.core.dataset import Dataset


def _populate_compressed_samples(tensor: Tensor, cat_path, flower_path, count=1):
    original_compressions = []

    for _ in range(count):
        tensor.append(hub.read(cat_path))
        original_compressions.append("jpg")

        tensor.append(hub.read(flower_path))
        original_compressions.append("png")

        tensor.append(np.ones((100, 100, 4), dtype="uint8"))
        original_compressions.append(tensor.meta.sample_compression)

        tensor.extend(
            [
                hub.read(flower_path),
                hub.read(cat_path),
            ]
        )
        original_compressions.extend(["png", "jpeg"])

    return original_compressions


@enabled_datasets
def test_populate_compressed_samples(ds: Dataset, cat_path, flower_path):
    images = ds.create_tensor(TENSOR_KEY, htype="image", sample_compression="png")

    assert images.meta.dtype == "uint8"
    assert images.meta.sample_compression == "png"

    _populate_compressed_samples(images, cat_path, flower_path)

    assert images[0].numpy().shape == (900, 900, 3)
    assert images[1].numpy().shape == (513, 464, 4)

    assert len(images) == 5
    assert images.shape == (5, None, None, None)
    assert images.shape_interval.lower == (5, 100, 100, 3)
    assert images.shape_interval.upper == (5, 900, 900, 4)


@enabled_datasets
def test_iterate_compressed_samples(ds: Dataset, cat_path, flower_path):
    images = ds.create_tensor(TENSOR_KEY, htype="image", sample_compression="png")

    assert images.meta.dtype == "uint8"
    assert images.meta.sample_compression == "png"

    _populate_compressed_samples(images, cat_path, flower_path)

    expected_shapes = [
        (900, 900, 3),
        (513, 464, 4),
        (100, 100, 4),
        (513, 464, 4),
        (900, 900, 3),
    ]

    assert len(images) == len(expected_shapes)
    for image, expected_shape in zip(images, expected_shapes):
        x = image.numpy()

        assert (
            type(x) == np.ndarray
        ), "Check is necessary in case a `PIL` object is returned instead of an array."
        assert x.shape == expected_shape


@enabled_datasets
def test_uncompressed(ds: Dataset):
    images = ds.create_tensor(TENSOR_KEY, sample_compression=None)

    images.append(np.ones((100, 100, 100)))
    images.extend(np.ones((3, 101, 2, 1)))
    ds.clear_cache()
    np.testing.assert_array_equal(images[0].numpy(), np.ones((100, 100, 100)))
    np.testing.assert_array_equal(images[1:4].numpy(), np.ones((3, 101, 2, 1)))


@pytest.mark.xfail(raises=SampleCompressionError, strict=True)
@pytest.mark.parametrize(
    "bad_shape",
    [
        # raises OSError: cannot write mode LA as JPEG
        (100, 100, 2),
        # raises OSError: cannot write mode RGBA as JPE
        (100, 100, 4),
    ],
)
def test_jpeg_bad_shapes(memory_ds: Dataset, bad_shape):
    # jpeg allowed shapes:
    # ---------------------
    # (100) works!
    # (100,) works!
    # (100, 100) works!
    # (100, 100, 1) works!
    # (100, 100, 2) raises   | OSError: cannot write mode LA as JPEG
    # (100, 100, 3) works!
    # (100, 100, 4) raises   | OSError: cannot write mode RGBA as JPEG
    # (100, 100, 5) raises   | TypeError: Cannot handle this data type: (1, 1, 5), |u1
    # (100, 100, 100) raises | TypeError: Cannot handle this data type: (1, 1, 100), |u1

    tensor = memory_ds.create_tensor(TENSOR_KEY, sample_compression="jpeg")
    tensor.append(np.ones(bad_shape, dtype="uint8"))


def test_compression_aliases(memory_ds: Dataset):
    tensor = memory_ds.create_tensor("jpeg_tensor", sample_compression="jpeg")
    assert tensor.meta.sample_compression == "jpeg"

    tensor = memory_ds.create_tensor("jpg_tensor", sample_compression="jpg")
    assert tensor.meta.sample_compression == "jpeg"


@pytest.mark.xfail(raises=UnsupportedCompressionError, strict=True)
def test_unsupported_compression(memory_ds: Dataset):
    memory_ds.create_tensor(TENSOR_KEY, sample_compression="bad_compression")
    # TODO: same tests but with `dtype`


@pytest.mark.xfail(raises=TensorMetaMissingRequiredValue, strict=True)
def test_missing_sample_compression_for_image(memory_ds: Dataset):
    memory_ds.create_tensor("tensor", htype="image")
