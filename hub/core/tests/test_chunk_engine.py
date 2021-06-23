from copy import deepcopy
from hub.core.index.index import Index
import numpy as np
from hub.core.storage.provider import StorageProvider
from hub.core._chunk_engine import ChunkEngine

from hub.constants import MB, KB


KEY = "chunks"


def test_scalars(memory_storage: StorageProvider):
    engine = ChunkEngine(KEY, memory_storage)

    engine.extend(np.arange(1000))
    engine.append(1000)
    engine.append(1001)

    for i in range(1002):
        assert engine.get_sample(Index(i)) == i

    assert engine.num_chunks == 1


def test_fixed_arrays(memory_storage: StorageProvider):
    engine = ChunkEngine(
        KEY,
        memory_storage,
        None,
        min_chunk_size_target=10 * KB,
        max_chunk_size=100 * KB,
    )

    a1 = np.arange(10 * 28 * 28 * 3, dtype=np.int32).reshape(10, 28, 28, 3)
    a1_copy = deepcopy(a1)

    # add approximately 95KB
    engine.extend(a1)
    engine.append(a1[2])
    engine.append(a1[-1])

    np.testing.assert_array_equal(a1_copy, engine.get_sample(Index(slice(0, 10))))
    np.testing.assert_array_equal(a1_copy[2], engine.get_sample(Index(10)))
    np.testing.assert_array_equal(a1_copy[-1], engine.get_sample(Index(-1)))

    assert engine.num_chunks == 1
    assert engine.num_samples == 12

    # add approximately 94KB (requires 2 chunks)
    engine.extend(a1)

    np.testing.assert_array_equal(a1_copy[-1], engine.get_sample(Index(-1)))

    assert engine.num_chunks == 2
    assert engine.num_samples == 22


def test_dynamic_arrays(memory_storage: StorageProvider):
    # TODO: dynamic samples that fit multiple in 1 chunk

    # TODO: dynamic samples that are too large for 1 chunk
    pass


def test_calculate_bytes():
    a1 = np.arange(10 * 28 * 28 * 3, dtype=np.int32).reshape(10, 28, 28, 3)
    expected_bytes = ChunkEngine.calculate_bytes((10, 28, 28, 3), np.int32)
    assert a1.nbytes == expected_bytes


def test_failures(memory_storage: StorageProvider):
    # TODO: index error and stuff
    pass
