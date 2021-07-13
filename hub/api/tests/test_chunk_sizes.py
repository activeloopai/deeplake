import numpy as np
from hub.constants import KB
from hub.core.tests.common import parametrize_all_dataset_storages


@parametrize_all_dataset_storages
def test_chunk_sizes(ds):
    images = ds.create_tensor("images", htype="image", sample_compression=None)
    labels = ds.create_tensor("labels", htype="class_label")

    images_engine = images.chunk_engine
    labels_engine = labels.chunk_engine

    # TODO: set / update chunk size API
    # set chunk sizes to be small (no API for this, so we have to explicitly set)
    max_chunk_size = 32 * KB
    min_chunk_size = max_chunk_size // 2
    images_engine.max_chunk_size = max_chunk_size
    images_engine.min_chunk_size = min_chunk_size
    labels_engine.max_chunk_size = max_chunk_size
    labels_engine.min_chunk_size = min_chunk_size

    n = 100

    for i in range(n):
        x = np.ones((28, 28), dtype=np.uint8) * i
        y = np.uint32(i)

        images.append(x)
        labels.append(y)

    # check number of chunks for labels
    labels_chunk_ids = labels_engine.chunk_id_encoder
    assert labels_chunk_ids.num_chunks == 1

    # check number of chunks for images
    images_chunk_ids = images_engine.chunk_id_encoder
    assert images_chunk_ids.num_chunks == 3
