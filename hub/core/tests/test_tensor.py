import numpy as np

from hub.core.tensor import (
    append_tensor,
    create_tensor,
    extend_tensor,
    read_samples_from_tensor,
)
from hub.tests.common import TENSOR_KEY
from hub.core.typing import StorageProvider
from .common import parametrize_all_storages_and_caches


@parametrize_all_storages_and_caches
def test_fixed_shape(storage: StorageProvider):
    create_tensor(TENSOR_KEY, storage)

    a1 = np.arange(100).reshape(50, 2)
    a2 = np.arange(1000).reshape(10, 50, 2)
    append_tensor(a1, TENSOR_KEY, storage)
    extend_tensor(a2, TENSOR_KEY, storage)

    out = read_samples_from_tensor(TENSOR_KEY, storage)
    np.testing.assert_array_equal(out, [a1, *a2])


@parametrize_all_storages_and_caches
def test_dynamic_shape(storage: StorageProvider):
    create_tensor(TENSOR_KEY, storage)

    a1 = np.arange(100).reshape(25, 4)
    a2 = np.arange(1000).reshape(10, 50, 2)
    a3 = np.arange(10).reshape(10, 1)
    a4 = np.arange(100).reshape(1, 100)
    a5 = np.arange(1000).reshape(10, 1, 100)
    append_tensor(a1, TENSOR_KEY, storage)
    extend_tensor(a2, TENSOR_KEY, storage)
    append_tensor(a3, TENSOR_KEY, storage)
    append_tensor(a4, TENSOR_KEY, storage)
    extend_tensor(a5, TENSOR_KEY, storage)

    out = read_samples_from_tensor(TENSOR_KEY, storage, aslist=True)
    expected_out = [a1, *a2, a3, a4, *a5]

    assert len(out) == 23
    for actual, expected in zip(out, expected_out):
        np.testing.assert_array_equal(actual, expected)
