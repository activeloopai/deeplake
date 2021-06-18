from hub.api.tensor import Tensor
from hub.tests.common import TENSOR_KEY, assert_all_samples_have_expected_compression
from hub.constants import UNCOMPRESSED
import numpy as np

import hub
from hub import Dataset
from hub.core.tests.common import parametrize_all_dataset_storages


def _populate_compressed_samples(tensor: Tensor, cat_path, flower_path, count=1):
    for _ in range(count):
        tensor.append(hub.load(cat_path))
        tensor.append(hub.load(flower_path))
        tensor.append(np.ones((100, 100, 4), dtype="uint8"))

        tensor.extend(
            [
                hub.load(flower_path),
                hub.load(cat_path),
            ]
        )


@parametrize_all_dataset_storages
def test_populate_compressed_samples(ds: Dataset, cat_path, flower_path):
    images = ds.create_tensor(TENSOR_KEY, htype="image")

    assert images.meta.dtype == "uint8"
    assert images.meta.sample_compression == "png"
    assert images.meta.chunk_compression == UNCOMPRESSED

    _populate_compressed_samples(images, cat_path, flower_path)
    assert_all_samples_have_expected_compression(images)

    assert images[0].numpy().shape == (900, 900, 3)
    assert images[1].numpy().shape == (513, 464, 4)

    assert len(images) == 5
    assert images.shape == (5, None, None, None)
    assert images.shape_interval.lower == (5, 100, 100, 3)
    assert images.shape_interval.upper == (5, 900, 900, 4)


@parametrize_all_dataset_storages
def test_iterate_compressed_samples(ds: Dataset, cat_path, flower_path):
    images = ds.create_tensor(TENSOR_KEY, htype="image")

    assert images.meta.dtype == "uint8"
    assert images.meta.sample_compression == "png"
    assert images.meta.chunk_compression == UNCOMPRESSED

    _populate_compressed_samples(images, cat_path, flower_path)

    assert_all_samples_have_expected_compression(images)

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
