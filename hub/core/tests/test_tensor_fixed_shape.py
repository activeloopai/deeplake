from hub.core.storage.memory import MemoryProvider
from typing import Tuple

import numpy as np
import pytest

from hub.core.tests.common import (
    parametrize_all_storages_and_caches,
)
from hub.core.tests.common import run_engine_test
from hub.core.typing import StorageProvider
from hub.tests.common import (
    SHAPE_PARAM,
    parametrize_chunk_sizes,
    parametrize_num_batches,
    parametrize_dtypes,
    get_random_array,
)
from hub.core.tests.common import (
    parametrize_all_storages_and_caches,
)

np.random.seed(1)

UNBATCHED_SHAPES = (
    (1,),
    (100,),
    (1, 1, 3),
    (20, 90),
    (3, 28, 24, 1),
)

BATCHED_SHAPES = (
    (1, 1),
    (10, 1),
    (1, 30, 30),
    (3, 3, 12, 12, 1),
)


@pytest.mark.parametrize(SHAPE_PARAM, UNBATCHED_SHAPES)
@parametrize_num_batches
@parametrize_chunk_sizes
@parametrize_dtypes
def test_unbatched(
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    memory_storage: MemoryProvider,
):
    """
    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITHOUT a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]
    run_engine_test(arrays, memory_storage, batched=False, chunk_size=chunk_size)


@pytest.mark.parametrize(SHAPE_PARAM, BATCHED_SHAPES)
@parametrize_num_batches
@parametrize_chunk_sizes
@parametrize_dtypes
def test_batched(
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    memory_storage: MemoryProvider,
):
    """
    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]
    run_engine_test(arrays, memory_storage, batched=True, chunk_size=chunk_size)
