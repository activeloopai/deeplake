from hub.core.chunk_engine.write import write_array
import os
from hub.tests.common import get_dummy_data_path
from hub.util.dataset import get_compressor
from PIL import Image
import numpy as np
import pytest

import hub
from hub import Dataset
from hub.core.tests.common import parametrize_all_dataset_storages


@parametrize_all_dataset_storages
def test_load_compressed_samples(ds: Dataset):
    # TODO: make test fixtures for these paths
    path = get_dummy_data_path("compressed_images")
    cat_path = os.path.join(path, "cat.jpeg")
    flower_path = os.path.join(path, "flower.png")

    images = ds.create_tensor("images", htype="image")

    # TODO: test symbolic load + load in isolation
    images.append(hub.load(cat_path))
    images.append(hub.load(flower_path))

    images.extend(
        [
            hub.load(flower_path),
            hub.load(cat_path),
        ]
    )

    assert images[0].numpy().shape == (900, 900, 3)
    assert images[1].numpy().shape == (513, 464, 4)

    assert len(images) == 4
    assert images.shape == (4, None, None, None)
    assert images.shape_interval.lower == (4, 513, 464, 3)
    assert images.shape_interval.upper == (4, 900, 900, 4)
