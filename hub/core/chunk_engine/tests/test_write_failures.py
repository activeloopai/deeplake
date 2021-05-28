import pytest
import numpy as np

from hub.util.exceptions import (
    MetaMismatchError,
)
from hub.core.chunk_engine import add_samples_to_tensor
from hub.tests.common import TENSOR_KEY


@pytest.mark.xfail(raises=MetaMismatchError, strict=True)
def test_dtype_mismatch(memory_storage):
    a1 = np.array([1, 2, 3, 5.3], dtype=float)
    a2 = np.array([0, 1, 1, 0], dtype=bool)
    add_samples_to_tensor(a1, TENSOR_KEY, memory_storage, batched=False)
    add_samples_to_tensor(a2, TENSOR_KEY, memory_storage, batched=False)


@pytest.mark.xfail(raises=MetaMismatchError, strict=True)
def test_shape_length_mismatch(memory_storage):
    a1 = np.arange(100).reshape(5, 20)
    a2 = np.arange(200).reshape(5, 20, 2)
    add_samples_to_tensor(a1, TENSOR_KEY, memory_storage, batched=False)
    add_samples_to_tensor(a2, TENSOR_KEY, memory_storage, batched=False)
