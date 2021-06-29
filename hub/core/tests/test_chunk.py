from copy import deepcopy
import numpy as np

from hub.core.chunk import Chunk, chunk_from_buffer
from hub.constants import KB


def _assert_buffer_recomposition(chunk: Chunk):
    chunk_buffer = chunk.tobytes()
    frombuffer_chunk = chunk_from_buffer(chunk_buffer)
    assert chunk_buffer == frombuffer_chunk.tobytes()


def test_single_chunk():
    chunk = Chunk(max_data_bytes=10 * KB)

    num_samples = 1
    shape = (100, 100, 3)
    a = np.ones((num_samples, *shape), dtype=np.int64)
    a_copy = deepcopy(a)
    a_buffer = a.tobytes()

    new_chunks = chunk.extend(a_buffer, num_samples, shape)
    np.testing.assert_array_equal(chunk.numpy(), a_copy)

    assert len(new_chunks) == 0
    assert chunk.num_samples == 1

    _assert_buffer_recomposition(chunk)


def test_scalars():
    chunk = Chunk(max_data_bytes=8 * KB)

    a = np.ones(1000)

    new_chunks = chunk.extend(a.tobytes(), 5000, (1,))

    assert len(new_chunks) == 4
    assert chunk.num_samples == 1000

    np.testing.assert_array_equal(chunk.numpy(), a)

    _assert_buffer_recomposition(chunk)
