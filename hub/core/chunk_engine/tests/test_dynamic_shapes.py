import pytest

import numpy as np

from hub.core.chunk_engine.dummy_util import (
    MemoryProvider,
    DummySampleCompression,
    DummyChunkCompression,
)

from hub.core.chunk_engine.tests.util import run_engine_test, CHUNK_SIZES


UNBATCHED_DYNAMIC_SHAPES = (
    [(1,), (2,), (3,)],
    [(3,), (2,), (1,)],
    [(100,), (99,), (1000,)],
    [(100, 100), (1, 1), (200, 200), (50, 50)],
    [(100, 100, 3), (100, 100, 1), (100, 100, 3), (10, 10, 1), (10, 10, 3)],
    [(3, 100, 100), (1, 100, 100), (3, 100, 100), (1, 10, 10), (3, 10, 10)],
    [(3, 100, 100, 1), (3, 100, 100, 1), (3, 100, 100, 1), (3, 100, 100, 2)],
)


BATCHED_DYNAMIC_SHAPES = (
    [(1, 1), (1, 2), (1, 3)],
    [(1, 1), (2, 2), (3, 3)],
    [(1, 3), (1, 2), (1, 1)],
    [(3, 3), (2, 2), (1, 1)],
    [(1, 100), (2, 99), (3, 1000)],
    [(3, 100), (2, 99), (1, 1000)],
    [
        (10, 3, 100, 100, 1),
        (1, 3, 100, 100, 1),
        (5, 3, 100, 100, 1),
        (1, 3, 100, 100, 2),
    ],
)


@pytest.mark.parametrize("shapes", UNBATCHED_DYNAMIC_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_unbatched(shapes, chunk_size):
    """
    Samples have DYNAMIC shapes (must have the same len(shape)).
    Samples are provided WITHOUT a batch axis.
    """

    storage = MemoryProvider()
    compression = DummyChunkCompression()

    arrays = [np.random.uniform(size=shape) for shape in shapes]

    run_engine_test(arrays, storage, compression, batched=False, chunk_size=chunk_size)


@pytest.mark.parametrize("shapes", BATCHED_DYNAMIC_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_batched(shapes, chunk_size):
    """
    Samples have DYNAMIC shapes (must have the same len(shape)).
    Samples are provided WITH a batch axis.
    """

    storage = MemoryProvider()
    compression = DummyChunkCompression()

    arrays = [np.random.uniform(size=shape) for shape in shapes]

    run_engine_test(arrays, storage, compression, batched=True, chunk_size=chunk_size)
