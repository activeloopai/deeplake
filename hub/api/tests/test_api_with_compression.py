from hub.core.meta.index_meta import IndexMeta
import os
from hub.tests.common import get_dummy_data_path
import numpy as np

import hub
from hub import Dataset
from hub.core.tests.common import parametrize_all_dataset_storages


def _get_compression_for_sample(ds: Dataset, tensor_name: str, idx: int) -> str:
    return IndexMeta.load(tensor_name, ds.storage).entries[idx]["compression"]


@parametrize_all_dataset_storages
def test_load_compressed_samples(ds: Dataset):
    # TODO: make test fixtures for these paths
    path = get_dummy_data_path("compressed_images")
    cat_path = os.path.join(path, "cat.jpeg")
    flower_path = os.path.join(path, "flower.png")

    images = ds.create_tensor("images", htype="image")

    assert images.meta.default_compression == "PNG"

    images.append(hub.load(cat_path))
    images.append(hub.load(flower_path))
    images.append(np.ones((100, 100, 4), dtype="uint8"))

    images.extend(
        [
            hub.load(flower_path),
            hub.load(cat_path),
        ]
    )

    # TODO: better way to check a sample's compression (in API)
    # TODO: also, maybe we should check if these bytes are ACTUALLY compressed. right now technically all of these compressions could just be identites
    assert _get_compression_for_sample(ds, "images", 0) == "JPEG"
    assert _get_compression_for_sample(ds, "images", 1) == "PNG"
    assert (
        _get_compression_for_sample(ds, "images", 2) == "PNG"
    ), "The default compression for `image` htypes is 'PNG'"

    assert images[0].numpy().shape == (900, 900, 3)
    assert images[1].numpy().shape == (513, 464, 4)

    assert len(images) == 5
    assert images.shape == (5, None, None, None)
    assert images.shape_interval.lower == (5, 100, 100, 3)
    assert images.shape_interval.upper == (5, 900, 900, 4)
