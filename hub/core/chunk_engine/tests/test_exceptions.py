import pytest
import numpy as np

from hub.util.exceptions import (
    KeyDoesNotExistError,
    KeyAlreadyExistsError,
    MetaMismatchError,
)
from hub.core.chunk_engine import write_array, append_array
from hub.tests.common import TENSOR_KEY


@pytest.mark.xfail(raises=KeyDoesNotExistError, strict=True)
def test_non_existent_tensor_append(memory_storage):
    append_array(np.array([1, 2, 3, 4]), TENSOR_KEY, memory_storage, batched=False)


@pytest.mark.xfail(raises=KeyAlreadyExistsError, strict=True)
def test_existing_tensor_write(memory_storage):
    write_array(np.array([1, 2, 3, 4]), TENSOR_KEY, memory_storage, batched=False)
    write_array(np.array([5, 6, 4]), TENSOR_KEY, memory_storage, batched=False)


@pytest.mark.xfail(raises=MetaMismatchError, strict=True)
def test_dtype_mismatch(memory_storage):
    a1 = np.array([1, 2, 3, 5.3], dtype=float)
    a2 = np.array([0, 1, 1, 0], dtype=bool)
    write_array(a1, TENSOR_KEY, memory_storage, batched=False)
    append_array(a2, TENSOR_KEY, memory_storage, batched=False)


@pytest.mark.xfail(raises=MetaMismatchError, strict=True)
def test_shape_length_mismatch(memory_storage):
    a1 = np.arange(100).reshape(5, 20)
    a2 = np.arange(200).reshape(5, 20, 2)
    write_array(a1, TENSOR_KEY, memory_storage, batched=False)
    append_array(a2, TENSOR_KEY, memory_storage, batched=False)
