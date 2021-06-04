from hub.util.index import Index
import numpy as np
import pytest

from hub.core.meta.tensor_meta import tensor_meta_from_array
from hub.core.tensor import add_samples_to_tensor, create_tensor, read_samples_from_tensor

from hub.tests.common import TENSOR_KEY
from hub.util.exceptions import (
    DynamicTensorNumpyError,
    TensorAlreadyExistsError,
    TensorDoesNotExistError,
    TensorMetaInvalidValue,
    TensorMetaMismatchError,
)


@pytest.mark.xfail(raises=TensorMetaMismatchError, strict=True)
def test_dtype_mismatch(memory_storage):
    a1 = np.array([1, 2, 3, 5.3], dtype=float)
    a2 = np.array([0, 1, 1, 0], dtype=bool)
    create_tensor(TENSOR_KEY, memory_storage, tensor_meta_from_array(a1, batched=False))
    add_samples_to_tensor(a1, TENSOR_KEY, memory_storage, batched=False)
    add_samples_to_tensor(a2, TENSOR_KEY, memory_storage, batched=False)


@pytest.mark.xfail(raises=TensorMetaMismatchError, strict=True)
def test_shape_length_mismatch(memory_storage):
    a1 = np.arange(3 * 15).reshape(3, 15)
    a2 = np.arange(5 * 20 * 2).reshape(5, 20, 2)
    create_tensor(TENSOR_KEY, memory_storage, tensor_meta_from_array(a1, batched=False))
    add_samples_to_tensor(a1, TENSOR_KEY, memory_storage, batched=False)
    add_samples_to_tensor(a2, TENSOR_KEY, memory_storage, batched=False)


@pytest.mark.xfail(raises=TensorDoesNotExistError, strict=True)
def test_tensor_does_not_exist(memory_storage):
    a1 = np.arange(10)
    add_samples_to_tensor(a1, TENSOR_KEY, memory_storage, batched=False)


@pytest.mark.xfail(raises=TensorAlreadyExistsError, strict=True)
def test_tensor_already_exists(memory_storage):
    a1 = np.arange(10)
    create_tensor(TENSOR_KEY, memory_storage, tensor_meta_from_array(a1, batched=False))
    create_tensor(TENSOR_KEY, memory_storage, tensor_meta_from_array(a1, batched=False))


@pytest.mark.xfail(raises=TensorMetaInvalidValue, strict=True)
@pytest.mark.parametrize("chunk_size", [0, -1, -100])
def test_invalid_chunk_sizes(memory_storage, chunk_size):
    create_tensor(
        TENSOR_KEY, memory_storage, {"dtype": "int", "chunk_size": chunk_size}
    )


@pytest.mark.xfail(raises=TensorMetaInvalidValue, strict=True)
@pytest.mark.parametrize("dtype", [1, False, "floatf", "intj", "foo", "bar"])
def test_invalid_dtypes(memory_storage, dtype):
    create_tensor(TENSOR_KEY, memory_storage, {"dtype": dtype, "chunk_size": 4})


@pytest.mark.xfail(raises=DynamicTensorNumpyError, strict=True)
def test_dynamic_as_numpy(memory_storage):
    a1 = np.ones((9, 23))
    a2 = np.ones((99, 2))
    create_tensor(TENSOR_KEY, memory_storage, {"dtype": "float64", "chunk_size": 4096})
    add_samples_to_tensor(a1, TENSOR_KEY, memory_storage, batched=False)
    add_samples_to_tensor(a2, TENSOR_KEY, memory_storage, batched=False)

    # aslist=False, but a1 / a2 are not the same shape
    read_samples_from_tensor(TENSOR_KEY, memory_storage, aslist=False)