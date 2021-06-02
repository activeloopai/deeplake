from typing import Tuple, List

import numpy as np
import pytest
from hub.core.tests.common import run_engine_test
from hub.tests.common import (
    SHAPE_PARAM,
    parametrize_chunk_sizes,
    parametrize_dtypes,
    get_random_array
)
from hub.core.tests.common import (
    parametrize_all_storages_and_caches,
)
from hub.core.typing import StorageProvider

np.random.seed(1)


SHAPE_PARAM = "shapes"


# len(UNBATCHED_SHAPES)[i] must be > 1
UNBATCHED_SHAPES = (
    [(1,), (5,)],
    [(100,), (1,)],
    [(1, 1, 3), (1,), (5,), (3,)],
    [(20, 90), (25, 2), (2, 2), (10, 10, 1)],
    [(3, 28, 24, 1), (2, 22, 25, 1)],
)

# len(BATCHED_SHAPES)[i] must be > 1
BATCHED_SHAPES = (
    [(1, 1), (3, 5)],
    [(10, 1), (10, 2)],
    [(1, 30, 30), (1, 2, 2), (5, 3, 50)],
    [(3, 3, 12, 12, 1), (1, 3, 12, 15), (1, 6, 5, 3, 1, 1, 1, 1)],
)


@pytest.mark.parametrize(SHAPE_PARAM, UNBATCHED_SHAPES)
@parametrize_chunk_sizes
@parametrize_dtypes
@parametrize_all_storages_and_caches
def test_unbatched(
    shapes: List[Tuple[int]],
    chunk_size: int,
    dtype: str,
    storage: StorageProvider,
):
    """
    Samples have DYNAMIC shapes (can have different shapes).
    Samples are provided WITHOUT a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for shape in shapes]
    run_engine_test(arrays, storage, batched=False, chunk_size=chunk_size)


@pytest.mark.parametrize(SHAPE_PARAM, BATCHED_SHAPES)
@parametrize_chunk_sizes
@parametrize_dtypes
@parametrize_all_storages_and_caches
def test_batched(
    shapes: List[Tuple[int]],
    chunk_size: int,
    dtype: str,
    storage: StorageProvider,
):
    """
    Samples have DYNAMIC shapes (can have different shapes).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for shape in shapes]
    run_engine_test(arrays, storage, batched=True, chunk_size=chunk_size)
