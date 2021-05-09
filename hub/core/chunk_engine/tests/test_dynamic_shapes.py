import pytest

import numpy as np

from hub.core.chunk_engine.dummy_util import (
    DummySampleCompression,
    DummyChunkCompression,
)

from hub.core.chunk_engine.tests.util import (
    run_engine_test,
    CHUNK_SIZES,
    DTYPES,
    get_random_array,
    COMPRESSIONS,
    STORAGE_PROVIDERS,
)


np.random.seed(1)


# TODO: add failure tests (where len(shape) differs)


UNBATCHED_SHAPES = (
    [(1,), (2,), (3,)],
    [(100,), (99,), (1000,)],
    [(100, 100), (1, 1), (200, 200), (50, 50)],
    [(3, 100, 100, 1), (3, 100, 100, 1), (3, 100, 100, 1), (3, 100, 100, 2)],
)


BATCHED_SHAPES = (
    [(1, 1), (1, 2), (1, 3)],
    [(3, 3), (2, 2), (1, 1)],
    [
        (10, 3, 100, 100, 1),
        (1, 3, 100, 100, 1),
        (5, 3, 100, 100, 1),
        (1, 3, 100, 100, 2),
    ],
)


@pytest.mark.parametrize("shapes", UNBATCHED_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
def test_unbatched(shapes, chunk_size, dtype, compression, storage):
    """
    Samples have DYNAMIC shapes (must have the same len(shape)).
    Samples are provided WITHOUT a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for shape in shapes]

    run_engine_test(arrays, storage, compression, batched=False, chunk_size=chunk_size)


@pytest.mark.parametrize("shapes", BATCHED_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
def test_batched(shapes, chunk_size, dtype, compression, storage):
    """
    Samples have DYNAMIC shapes (must have the same len(shape)).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for shape in shapes]

    run_engine_test(arrays, storage, compression, batched=True, chunk_size=chunk_size)
