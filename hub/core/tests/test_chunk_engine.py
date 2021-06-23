from copy import deepcopy
from hub.core.index.index import Index
import numpy as np
from hub.core.storage.provider import StorageProvider
from hub.core._chunk_engine import ChunkEngine

from hub.constants import B, MB, KB


KEY = "chunks"


def test_scalars(memory_storage: StorageProvider):
    engine = ChunkEngine(KEY, memory_storage, create_tensor=True)

    engine.extend(np.arange(1000))
    engine.append(1000)
    engine.append(1001)

    for i in range(1002):
        assert engine.get_sample(Index(i)) == i

    # assert engine.num_chunks == 1  # TODO: uncomment me!


def test_arrays(memory_storage: StorageProvider):
    engine = ChunkEngine(
        KEY,
        memory_storage,
        min_chunk_size_target=1 * KB,
        max_chunk_size=5 * KB,
        create_tensor=True,
    )

    a1 = np.arange(3 * 10 * 10 * 3, dtype=np.int32).reshape(3, 10, 10, 3)
    assert a1.nbytes > engine.min_chunk_size_target
    assert a1.nbytes < engine.max_chunk_size
    assert a1.nbytes * 2 > engine.max_chunk_size
    a1_copy = deepcopy(a1)

    engine.extend(a1)
    engine.append(a1[0])
    engine.append(a1[-1])

    assert engine.num_samples == 5
    # assert engine.num_chunks == 1  # TODO: uncomment me!

    a2 = np.arange(3 * 9 * 11 * 4, dtype=np.int32).reshape(3, 9, 11, 4)
    assert a2.nbytes > engine.min_chunk_size_target
    assert a2.nbytes < engine.max_chunk_size
    assert a2.nbytes * 2 > engine.max_chunk_size
    a2_copy = deepcopy(a2)

    # requires 2 chunks to do this
    engine.extend(a2)

    actual_samples = engine.get_sample(Index(), aslist=True)
    expected_samples = [*a1_copy, a1_copy[0], a1_copy[-1], *a2_copy]

    for actual_sample, expected_sample in zip(actual_samples, expected_samples):
        np.testing.assert_array_equal(actual_sample, expected_sample)

    assert engine.num_samples == 8
    # assert engine.num_chunks == 2  # TODO: uncomment me!


def test_large_arrays(memory_storage: StorageProvider):
    engine = ChunkEngine(
        KEY,
        memory_storage,
        min_chunk_size_target=1 * KB,
        max_chunk_size=5 * KB,
        create_tensor=True,
    )

    a1 = np.arange(10 * 10 * 10 * 3, dtype=np.int32).reshape(10, 10, 10, 3)
    assert a1.nbytes > engine.max_chunk_size * 2 + engine.min_chunk_size_target
    assert a1.nbytes < engine.max_chunk_size * 3
    assert a1.nbytes * 2 > engine.max_chunk_size
    a1_copy = deepcopy(a1)

    engine.extend(a1)

    np.testing.assert_array_equal(a1_copy, engine.get_sample(Index(slice(0, 10))))

    assert engine.num_samples == 10
    # assert engine.num_chunks == 3   # TODO: uncomment me!

    a2 = np.arange(10 * 9 * 10 * 4, dtype=np.int32).reshape(10, 9, 10, 4)
    assert a2.nbytes > engine.max_chunk_size * 2 + engine.min_chunk_size_target
    assert a2.nbytes < engine.max_chunk_size * 3
    assert a2.nbytes * 2 > engine.max_chunk_size
    a2_copy = deepcopy(a2)

    engine.extend(a2)

    np.testing.assert_array_equal(a2_copy, engine.get_sample(Index(slice(10, 20))))

    assert engine.num_samples == 20
    # assert engine.num_chunks == 6   # TODO: uncomment me!


def test_calculate_bytes():
    a1 = np.arange(10 * 28 * 28 * 3, dtype=np.int32).reshape(10, 28, 28, 3)
    expected_bytes = ChunkEngine.calculate_bytes((10, 28, 28, 3), np.int32)
    assert a1.nbytes == expected_bytes


def test_failures(memory_storage: StorageProvider):
    # TODO: index error and stuff
    # TODO: test that a dynamic tensor's len(shape) must be equal
    pass
