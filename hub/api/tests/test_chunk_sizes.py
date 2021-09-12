import numpy as np
from hub.constants import KB


def _create_tensors(ds):
    images = ds.create_tensor(
        "images", htype="image", sample_compression=None, max_chunk_size=32 * KB
    )
    labels = ds.create_tensor("labels", htype="class_label", max_chunk_size=32 * KB)
    return images, labels


def _append_tensors(images, labels):
    for i in range(100):
        x = np.ones((28, 28), dtype=np.uint8) * i
        y = np.uint32(i)

        images.append(x)
        labels.append(y)


def _extend_tensors(images, labels):
    images.extend(np.ones((100, 28, 28), dtype=np.uint8))
    labels.extend(np.ones(100, dtype=np.uint32))


def test_append(memory_ds):
    ds = memory_ds
    images, labels = _create_tensors(ds)

    _append_tensors(images, labels)

    assert labels.num_chunks == 1
    assert images.num_chunks == 5

    _append_tensors(images, labels)

    assert labels.num_chunks == 1
    assert images.num_chunks == 10

    _append_tensors(images, labels)

    assert labels.num_chunks == 1
    assert images.num_chunks == 15

    assert len(ds) == 300


def test_extend(memory_ds):
    ds = memory_ds
    images, labels = _create_tensors(ds)

    _extend_tensors(images, labels)

    assert labels.num_chunks == 1
    assert images.num_chunks == 5

    _extend_tensors(images, labels)

    assert labels.num_chunks == 1
    assert images.num_chunks == 10

    _extend_tensors(images, labels)

    assert labels.num_chunks == 1
    assert images.num_chunks == 15

    assert len(ds) == 300


def test_extend_and_append(memory_ds):
    ds = memory_ds
    images, labels = _create_tensors(ds)

    _extend_tensors(images, labels)

    assert labels.num_chunks == 1
    assert images.num_chunks == 5

    _append_tensors(images, labels)

    assert labels.num_chunks == 1
    assert images.num_chunks == 10

    _extend_tensors(images, labels)

    assert labels.num_chunks == 1
    assert images.num_chunks == 15

    _append_tensors(images, labels)

    assert labels.num_chunks == 1
    assert images.num_chunks == 20

    assert len(ds) == 400
