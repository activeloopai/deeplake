from copy import deepcopy
import numpy as np

from hub.core.chunk import Chunk
from hub.constants import KB


def test_single_chunk():
    chunk = Chunk(max_data_bytes=10 * KB)

    num_samples = 1
    shape = (100, 100, 3)
    a = np.ones((num_samples, *shape), dtype=np.int64)
    a_copy = deepcopy(a)
    a_buffer = a.tobytes()

    new_chunks = chunk.extend(a_buffer, num_samples, shape)
    np.testing.assert_array_equal(chunk[0], a_copy)  # TODO: chunk.__eq__

    assert len(new_chunks) == 0
    assert chunk.num_samples_including_partial == 1


def test_multi_chunk_small_samples():
    chunk = Chunk(max_data_bytes=10 * KB)

    # 6,272 bytes per sample
    shape = (28, 28)
    dtype = np.int64

    # 31,360 bytes for 5 samples
    num_samples = 5

    a = np.ones((num_samples, *shape), dtype=dtype)
    a_copy = deepcopy(a)
    a_buffer = a.tobytes()

    new_chunks = chunk.extend(a_buffer, num_samples, shape)

    # 4 chunks are needed to represent all samples
    assert len(new_chunks) == 3

    assert chunk.num_samples_including_partial == 2
    assert new_chunks[0].num_samples_including_partial == 2
    assert new_chunks[1].num_samples_including_partial == 2
    assert new_chunks[2].num_samples_including_partial == 1

    np.testing.assert_array_equal(chunk[0:5], a_copy)


def test_scalars():
    chunk = Chunk(max_data_bytes=8 * KB)

    a = np.ones(5000)  # fill 5 chunks perfectly

    new_chunks = chunk.extend(a.tobytes(), 5000, (1,))

    assert len(new_chunks) == 4
    assert chunk.num_samples_including_partial == 1000
    assert new_chunks[0].num_samples_including_partial == 1000
    assert new_chunks[1].num_samples_including_partial == 1000
    assert new_chunks[2].num_samples_including_partial == 1000
    assert new_chunks[3].num_samples_including_partial == 1000

    np.testing.assert_array_equal(chunk[0:5000], a)
