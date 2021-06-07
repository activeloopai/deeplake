import os
from typing import Tuple
from uuid import uuid1

import numpy as np
import pytest

from hub.constants import KB, MB

SESSION_ID = str(uuid1())

TENSOR_KEY = "tensor"

SHAPE_PARAM = "shape"
NUM_BATCHES_PARAM = "num_batches"
DTYPE_PARAM = "dtype"
CHUNK_SIZE_PARAM = "chunk_size"

NUM_BATCHES = (1, 5)

CHUNK_SIZES = (
    1 * KB,
    1 * MB,
    16 * MB,
)

DTYPES = (
    "uint8",
    "int64",
    "float64",
    "bool",
)

parametrize_chunk_sizes = pytest.mark.parametrize(CHUNK_SIZE_PARAM, CHUNK_SIZES)
parametrize_dtypes = pytest.mark.parametrize(DTYPE_PARAM, DTYPES)
parametrize_num_batches = pytest.mark.parametrize(NUM_BATCHES_PARAM, NUM_BATCHES)


def current_test_name() -> str:
    full_name = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]  # type: ignore
    test_file = full_name.split("::")[0].split("/")[-1].split(".py")[0]
    test_name = full_name.split("::")[1]
    output = os.path.join(test_file, test_name)
    return output


def get_random_array(shape: Tuple[int], dtype: str) -> np.ndarray:
    dtype = dtype.lower()

    if "int" in dtype:
        low = np.iinfo(dtype).min
        high = np.iinfo(dtype).max
        return np.random.randint(low=low, high=high, size=shape, dtype=dtype)

    if "float" in dtype:
        # `low`/`high` have to be `float16` instead of `dtype` because `np.random.uniform` only supports `float16`
        low = np.finfo("float16").min
        high = np.finfo("float16").max
        return np.random.uniform(low=low, high=high, size=shape).astype(dtype)

    if "bool" in dtype:
        a = np.random.uniform(size=shape)
        return a > 0.5

    raise ValueError("Dtype %s not supported." % dtype)


@parametrize_dtypes
@pytest.mark.parametrize(
    SHAPE_PARAM,
    (
        (100, 100),
        (1,),
        (1, 1, 1, 1, 1),
    ),
)
def test_get_random_array(shape: Tuple[int], dtype: str):
    array = get_random_array(shape, dtype)
    assert array.shape == shape
    assert array.dtype == dtype
