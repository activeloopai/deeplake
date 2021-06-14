from hub.core.storage.memory import MemoryProvider
from hub.core.index import Index
from hub.util.array import (
    normalize_and_batchify_array_shape,
    normalize_and_batchify_shape,
)
from hub.core.meta.tensor_meta import read_tensor_meta
from hub.core.tensor import (
    add_samples_to_tensor,
    create_tensor,
    read_samples_from_tensor,
)
from typing import Tuple, List

import numpy as np
import pytest
from hub.core.tests.common import run_engine_test
from hub.tests.common import (
    TENSOR_KEY,
    parametrize_chunk_sizes,
    parametrize_dtypes,
    get_random_array,
)
from hub.core.tests.common import (
    parametrize_all_storages_and_caches,
)
from hub.core.typing import StorageProvider

np.random.seed(1)


SHAPES_PARAM = "shapes"  # different than `SHAPE_PARAM` (plural)


# len(UNBATCHED_SHAPES[i]) must be > 1
UNBATCHED_SHAPES = (
    [(1,), (5,)],
    [(100,), (1,)],
    [(1, 1, 3), (1,), (5,), (3,)],
    [(20, 90), (25, 2), (2, 2), (10, 10, 1)],
    [(3, 28, 24, 1), (2, 22, 25, 1)],
)

# len(BATCHED_SHAPES[i]) must be > 1
BATCHED_SHAPES = (
    [(1, 1), (3, 5)],
    [(10, 1), (10, 2)],
    [(1, 30, 30), (1, 2, 2), (5, 3, 50)],
    [(3, 3, 12, 12, 1), (1, 3, 12, 15), (1, 6, 5, 3, 1, 1, 1, 1)],
)


@pytest.mark.parametrize(SHAPES_PARAM, UNBATCHED_SHAPES)
@parametrize_chunk_sizes
@parametrize_dtypes
def test_unbatched(
    shapes: List[Tuple[int]],
    chunk_size: int,
    dtype: str,
    memory_storage: MemoryProvider,
):
    """
    Samples have DYNAMIC shapes (can have different shapes).
    Samples are provided WITHOUT a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for shape in shapes]
    run_engine_test(arrays, memory_storage, batched=False, chunk_size=chunk_size)


@pytest.mark.parametrize(SHAPES_PARAM, BATCHED_SHAPES)
@parametrize_chunk_sizes
@parametrize_dtypes
def test_batched(
    shapes: List[Tuple[int]],
    chunk_size: int,
    dtype: str,
    memory_storage: MemoryProvider,
):
    """
    Samples have DYNAMIC shapes (can have different shapes).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for shape in shapes]
    run_engine_test(arrays, memory_storage, batched=True, chunk_size=chunk_size)
