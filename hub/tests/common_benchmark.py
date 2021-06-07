import pytest

from hub.constants import MB
from hub.tests.common import (
    CHUNK_SIZE_PARAM,
    SHAPE_PARAM,
    NUM_BATCHES_PARAM,
    DTYPE_PARAM,
)

# benchmark parametrizations
BENCHMARK_NUM_BATCHES = (1,)
BENCHMARK_DTYPES = ("int64",)
BENCHMARK_CHUNK_SIZES = (16 * MB,)

FULL_BENCHMARK_BATCHED_SHAPES = (
    # with int64/float64 = ~1GB
    (840, 224, 224, 3),
)

BENCHMARK_BATCHED_SHAPES = ((5, 224, 224, 3),)

# benchmark parametrize decorators
parametrize_benchmark_chunk_sizes = pytest.mark.parametrize(
    CHUNK_SIZE_PARAM, BENCHMARK_CHUNK_SIZES
)
parametrize_benchmark_dtypes = pytest.mark.parametrize(DTYPE_PARAM, BENCHMARK_DTYPES)
parametrize_benchmark_shapes = pytest.mark.parametrize(
    SHAPE_PARAM, BENCHMARK_BATCHED_SHAPES
)
parametrize_full_benchmark_shapes = pytest.mark.parametrize(
    SHAPE_PARAM, FULL_BENCHMARK_BATCHED_SHAPES
)
parametrize_benchmark_num_batches = pytest.mark.parametrize(
    NUM_BATCHES_PARAM, BENCHMARK_NUM_BATCHES
)
