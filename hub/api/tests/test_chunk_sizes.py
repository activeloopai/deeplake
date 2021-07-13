import numpy as np
from hub.constants import KB
from hub.core.tests.common import parametrize_all_dataset_storages


def _update_chunk_sizes(ds, max_chunk_size: int):
    """Updates all chunk sizes for tensors that already exist in `ds`. If
    more tensors are created after calling this method, those tensors will NOT have
    the same chunk size.
    """

    # TODO: set / update chunk sizes API (to replace this function)

    min_chunk_size = max_chunk_size // 2

    for tensor in ds.tensors.values():
        chunk_engine = tensor.chunk_engine

        chunk_engine.max_chunk_size = max_chunk_size
        chunk_engine.min_chunk_size = min_chunk_size


def _assert_num_chunks(tensor, expected_num_chunks):
    chunk_engine = tensor.chunk_engine
    actual_num_chunks = chunk_engine.chunk_id_encoder.num_chunks
    assert actual_num_chunks == expected_num_chunks


@parametrize_all_dataset_storages
def test_chunk_sizes(ds):
    images = ds.create_tensor("images", htype="image", sample_compression=None)
    labels = ds.create_tensor("labels", htype="class_label")

    _update_chunk_sizes(ds, 32 * KB)

    n = 100

    for i in range(n):
        x = np.ones((28, 28), dtype=np.uint8) * i
        y = np.uint32(i)

        images.append(x)
        labels.append(y)

    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 5)
