from hub.util.index import Index
from hub.util.array import normalize_and_batchify_array_shape, normalize_and_batchify_shape
from hub.core.meta.tensor_meta import read_tensor_meta
from hub.core.tensor import add_samples_to_tensor, create_tensor, read_samples_from_tensor
from typing import Tuple, List

import numpy as np
import pytest
from hub.core.tests.common import run_engine_test
from hub.tests.common import (
    SHAPE_PARAM,
    TENSOR_KEY,
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
    [(2, 4,), (3, 4,), (0, 4)],  # simulate bounding boxes where 1 sample has no bbox
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


# TODO: failure case where adding arrays with different `len(array.shape)`


def run_dynamic_tensor_test(shapes: Tuple[int], storage: StorageProvider, dtype: str, chunk_size: int, batched: bool):
    arrays = [get_random_array(shape, dtype) for shape in shapes]

    create_tensor(TENSOR_KEY, storage, {"dtype": dtype, "chunk_size": chunk_size})

    for array in arrays:
        add_samples_to_tensor(array, TENSOR_KEY, storage, batched=batched)
    
    normalized_sample_shapes = [normalize_and_batchify_shape(shape, batched=batched)[1:] for shape in shapes]
    expected_min_shape = min(normalized_sample_shapes)
    expected_max_shape = max(normalized_sample_shapes)

    actual_meta = read_tensor_meta(TENSOR_KEY, storage)

    assert actual_meta["min_shape"] == expected_min_shape
    assert actual_meta["max_shape"] == expected_max_shape

    for i, expected_array in enumerate(arrays):
        actual_array = read_samples_from_tensor(TENSOR_KEY, storage, Index(i))
        expected_array = normalize_and_batchify_array_shape(expected_array, batched=batched)[0]
        np.testing.assert_array_equal(actual_array, expected_array)