from hub.core.index import Index
import numpy as np
import pytest

from hub.core.tensor import (
    append_tensor,
    create_tensor,
    read_samples_from_tensor,
)

from hub.tests.common import TENSOR_KEY
from hub.util.exceptions import (
    DynamicTensorNumpyError,
    TensorAlreadyExistsError,
    TensorDoesNotExistError,
    TensorInvalidSampleShapeError,
    TensorMetaInvalidHtypeOverwriteValue,
    TensorDtypeMismatchError,
)


@pytest.mark.xfail(raises=TensorDtypeMismatchError, strict=True)
def test_dtype_mismatch(memory_storage):
    a1 = np.array([1, 2, 3, 5.3], dtype=float)
    a2 = np.array([0, 1, 1, 0], dtype=bool)
    create_tensor(TENSOR_KEY, memory_storage)
    append_tensor(a1, TENSOR_KEY, memory_storage)
    append_tensor(a2, TENSOR_KEY, memory_storage)


@pytest.mark.xfail(raises=TensorInvalidSampleShapeError, strict=True)
def test_shape_length_mismatch(memory_storage):
    a1 = np.ones((5, 20))
    a2 = np.ones((5, 20, 2))
    create_tensor(TENSOR_KEY, memory_storage)
    append_tensor(a1, TENSOR_KEY, memory_storage)
    append_tensor(a2, TENSOR_KEY, memory_storage)


@pytest.mark.xfail(raises=TensorDoesNotExistError, strict=True)
def test_tensor_does_not_exist(memory_storage):
    a1 = np.arange(10)
    append_tensor(a1, TENSOR_KEY, memory_storage)


@pytest.mark.xfail(raises=TensorAlreadyExistsError, strict=True)
def test_tensor_already_exists(memory_storage):
    create_tensor(TENSOR_KEY, memory_storage)
    create_tensor(TENSOR_KEY, memory_storage)


@pytest.mark.xfail(raises=TensorMetaInvalidHtypeOverwriteValue, strict=True)
@pytest.mark.parametrize("chunk_size", [0, -1, -100])
def test_invalid_chunk_sizes(memory_storage, chunk_size):
    create_tensor(TENSOR_KEY, memory_storage, chunk_size=chunk_size)


@pytest.mark.xfail(raises=TensorMetaInvalidHtypeOverwriteValue, strict=True)
@pytest.mark.parametrize("dtype", [1, False, "floatf", "intj", "foo", "bar"])
def test_invalid_dtypes(memory_storage, dtype):
    create_tensor(TENSOR_KEY, memory_storage, dtype=dtype)


@pytest.mark.xfail(raises=DynamicTensorNumpyError, strict=True)
def test_dynamic_as_numpy(memory_storage):
    a1 = np.ones((9, 23))
    a2 = np.ones((99, 2))
    create_tensor(TENSOR_KEY, memory_storage)
    append_tensor(a1, TENSOR_KEY, memory_storage)
    append_tensor(a2, TENSOR_KEY, memory_storage)

    # aslist=False, but a1 / a2 are not the same shape
    read_samples_from_tensor(TENSOR_KEY, memory_storage, aslist=False)
