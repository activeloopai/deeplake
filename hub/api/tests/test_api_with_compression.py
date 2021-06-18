from hub.core.chunk_engine.read import (
    get_actual_compression_from_index_entry,
)
from hub.tests.common import TENSOR_KEY
from hub.api.tensor import Tensor
from hub.constants import UNCOMPRESSED
from hub.core.meta.index_meta import IndexMeta
import numpy as np

import hub
from hub import Dataset
from hub.core.tests.common import parametrize_all_dataset_storages


def _assert_all_same_compression(tensor: Tensor):
    storage = tensor.storage

    index_meta = IndexMeta.load(TENSOR_KEY, storage)
    expected_compression = tensor.meta.sample_compression

    for index_entry in index_meta.entries:
        actual_compression = get_actual_compression_from_index_entry(
            TENSOR_KEY, storage, index_entry
        )
        assert actual_compression == expected_compression


def _populate_compressed_samples(ds, cat_path, flower_path):
    images = ds.create_tensor(TENSOR_KEY, htype="image")

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
    _assert_all_same_compression(images)

    assert images[0].numpy().shape == (900, 900, 3)
    assert images[1].numpy().shape == (513, 464, 4)

    assert len(images) == 5
    assert images.shape == (5, None, None, None)
    assert images.shape_interval.lower == (5, 100, 100, 3)
    assert images.shape_interval.upper == (5, 900, 900, 4)


@parametrize_all_dataset_storages
def test_iterate_compressed_samples(ds: Dataset, cat_path, flower_path):
    images = _populate_compressed_samples(ds, cat_path, flower_path)
    _assert_all_same_compression(images)

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


# TODO: uncomment this -- it should raise a good error
# @parametrize_all_dataset_storages
# def test_something(ds: Dataset):
#     tensor = ds.create_tensor(TENSOR_KEY, htype="image", dtype="float64")
#     a = np.random.uniform(size=(100, 100))
#     tensor.append(a)
#     np.testing.assert_array_equal(a, tensor.numpy())
#
