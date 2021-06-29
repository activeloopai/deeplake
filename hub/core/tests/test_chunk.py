import pytest
from copy import deepcopy
import numpy as np

from hub.core.chunk import Chunk, chunk_from_buffer
from hub.constants import KB


def _assert_buffer_recomposition(chunk: Chunk):
    chunk_buffer = chunk.tobytes()
    frombuffer_chunk = chunk_from_buffer(chunk_buffer)
    assert chunk_buffer == frombuffer_chunk.tobytes()


@pytest.mark.parametrize("max_data_bytes", [8 * KB, 10 * KB])
def test_single_chunk(max_data_bytes: int):
    chunk = Chunk(max_data_bytes=max_data_bytes)

    num_samples = 1
    shape = (28, 28)
    a = np.ones((num_samples, *shape), dtype=np.int64)
    a_copy = deepcopy(a)
    a_buffer = a.tobytes()

    new_chunks = chunk.extend(a_buffer, num_samples, shape)
    np.testing.assert_array_equal(chunk.numpy(), a_copy)

    assert len(new_chunks) == 0
    assert chunk.num_samples == 1
    assert chunk.has_space

    _assert_buffer_recomposition(chunk)


@pytest.mark.parametrize("max_data_bytes", [8 * KB, 9 * KB])
def test_scalars(max_data_bytes: int):
    chunk = Chunk(max_data_bytes=max_data_bytes)

    a = np.ones(1000)

    new_chunks = chunk.extend(a.tobytes(), 5000, (1,))

    assert len(new_chunks) == 4
    assert chunk.num_samples == 1000
    assert not chunk.has_space

    np.testing.assert_array_equal(chunk.numpy(), a)

    _assert_buffer_recomposition(chunk)
