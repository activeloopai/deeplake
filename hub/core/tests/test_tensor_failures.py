import numpy as np
import pytest

from hub.core.meta.tensor_meta import tensor_meta_from_array
from hub.core.tensor import add_samples_to_tensor, create_tensor

from hub.tests.common import TENSOR_KEY
from hub.util.exceptions import TensorMetaMismatchError


@pytest.mark.xfail(raises=TensorMetaMismatchError, strict=True)
def test_dtype_mismatch(memory_storage):
    a1 = np.array([1, 2, 3, 5.3], dtype=float)
    a2 = np.array([0, 1, 1, 0], dtype=bool)
    create_tensor(TENSOR_KEY, memory_storage, tensor_meta_from_array(a1, batched=False))
    add_samples_to_tensor(a1, TENSOR_KEY, memory_storage, batched=False)
    add_samples_to_tensor(a2, TENSOR_KEY, memory_storage, batched=False)


@pytest.mark.xfail(raises=TensorMetaMismatchError, strict=True)
def test_shape_length_mismatch(memory_storage):
    a1 = np.arange(100).reshape(5, 20)
    a2 = np.arange(200).reshape(5, 20, 2)
    create_tensor(TENSOR_KEY, memory_storage, tensor_meta_from_array(a1, batched=False))
    add_samples_to_tensor(a1, TENSOR_KEY, memory_storage, batched=False)
    add_samples_to_tensor(a2, TENSOR_KEY, memory_storage, batched=False)


# TODO: failure case where `create_tensor` is not used
# TODO: failure case where `create_tensor` is used twice for same key
# TODO: failure case where `chunk_size <= 0`