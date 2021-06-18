from hub.api.tensor import Tensor
from hub.constants import UNCOMPRESSED
from hub.core.meta.index_meta import IndexMeta
import os
import numpy as np

import hub
from hub import Dataset
from hub.core.tests.common import parametrize_all_dataset_storages


def _get_compression_for_sample(ds: Dataset, tensor_name: str, idx: int) -> str:
    return IndexMeta.load(tensor_name, ds.storage).entries[idx]["compression"]


def _populate_compressed_samples(ds, cat_path, flower_path):
    images = ds.create_tensor("images", htype="image")
    assert images.meta.sample_compression == "png"
    assert images.meta.chunk_compression == UNCOMPRESSED

    images.append(hub.load(cat_path))
    images.append(hub.load(flower_path))
    images.append(np.ones((100, 100, 4), dtype="uint8"))

    images.extend(
        [
            hub.load(flower_path),
            hub.load(cat_path),
        ]
    )

    return images


@parametrize_all_dataset_storages
def test_populate_compressed_samples(ds: Dataset, cat_path, flower_path):
    images = _populate_compressed_samples(ds, cat_path, flower_path)

    # TODO: better way to check a sample's compression (in API)
    # TODO: also, maybe we should check if these bytes are ACTUALLY compressed. right now technically all of these compressions could just be identites
    assert _get_compression_for_sample(ds, "images", 0) == "jpeg"
    assert _get_compression_for_sample(ds, "images", 1) == "png"
    assert (
        _get_compression_for_sample(ds, "images", 2) == "png"
    ), "The default compression for `image` htypes is 'png'"

    assert images[0].numpy().shape == (900, 900, 3)
    assert images[1].numpy().shape == (513, 464, 4)

    assert len(images) == 5
    assert images.shape == (5, None, None, None)
    assert images.shape_interval.lower == (5, 100, 100, 3)
    assert images.shape_interval.upper == (5, 900, 900, 4)


@parametrize_all_dataset_storages
def test_iterate_compressed_samples(ds: Dataset, cat_path, flower_path):
    images = _populate_compressed_samples(ds, cat_path, flower_path)

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
