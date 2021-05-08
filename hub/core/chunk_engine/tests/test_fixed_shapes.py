import pytest

import numpy as np

from hub.core.chunk_engine.dummy_util import (
    MemoryProvider,
    DummySampleCompression,
    DummyChunkCompression,
)

from hub.core.chunk_engine.tests.util import (
    run_engine_test,
    CHUNK_SIZES,
    DTYPES,
    get_random_array,
)


np.random.seed(1)


# TODO: add failure tests (where shape differs)


# number of batches (unbatched implicitly = 1 sample per batch) per test
NUM_BATCHES = (
    1,
    5,
)


UNBATCHED_SHAPES = (
    (1,),
    (100,),
    (1, 1, 3),
    (100, 100),
    (3, 100, 100, 1),
)


BATCHED_SHAPES = (
    (1, 1),
    (10, 1),
    (1, 100, 100),
    (10, 3, 100, 100, 1),
)


@pytest.mark.parametrize("shape", UNBATCHED_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("num_batches", NUM_BATCHES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unbatched(shape, chunk_size, num_batches, dtype):
    """
    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITHOUT a batch axis.
    """

    storage = MemoryProvider()
    compression = DummyChunkCompression()

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]

    run_engine_test(arrays, storage, compression, batched=False, chunk_size=chunk_size)


@pytest.mark.parametrize("shape", BATCHED_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("num_batches", NUM_BATCHES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_batched(shape, chunk_size, num_batches, dtype):
    """
    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITH a batch axis.
    """

    storage = MemoryProvider()
    compression = DummyChunkCompression()

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]

    run_engine_test(arrays, storage, compression, batched=True, chunk_size=chunk_size)
