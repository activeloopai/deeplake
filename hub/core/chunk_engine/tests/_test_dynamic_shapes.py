import pytest

import numpy as np

from hub.core.chunk_engine.tests.common import (
    run_engine_test,
    CHUNK_SIZES,
    DTYPES,
    get_random_array,
    STORAGE_PROVIDERS,
)


np.random.seed(1)


UNBATCHED_SHAPES = (
    [(1,), (2,), (3,)],
    [(100,), (99,), (1000,)],
    [(19, 18), (1, 1), (1, 200), (10, 50)],
    [(3, 30, 16, 1), (3, 16, 1, 1), (3, 9, 9, 1), (3, 30, 16, 2)],
)


BATCHED_SHAPES = (
    [(1, 1), (1, 2), (1, 3)],
    [(3, 3), (2, 2), (1, 1)],
    [
        (10, 3, 24, 23, 1),
        (1, 3, 13, 13, 1),
        (5, 3, 25, 26, 1),
        (1, 3, 1, 19, 2),
    ],
)


@pytest.mark.parametrize("shapes", UNBATCHED_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
def test_unbatched(shapes, chunk_size, dtype, storage):
    """
    Samples have DYNAMIC shapes (must have the same len(shape)).
    Samples are provided WITHOUT a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for shape in shapes]
    run_engine_test(arrays, storage, batched=False, chunk_size=chunk_size)


@pytest.mark.parametrize("shapes", BATCHED_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
def test_batched(shapes, chunk_size, dtype, storage):
    """
    Samples have DYNAMIC shapes (must have the same len(shape)).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for shape in shapes]
    run_engine_test(arrays, storage, batched=True, chunk_size=chunk_size)
