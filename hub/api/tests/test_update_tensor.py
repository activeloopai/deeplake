from hub.api.tensor import Tensor
from hub.util.keys import get_index_meta_key, get_tensor_meta_key
import numpy as np

from hub.core.storage.provider import StorageProvider
import pytest
from hub.core.tensor import read_samples_from_tensor, tensor_exists
from hub.util.exceptions import TensorDoesNotExistError
from hub.api.dataset import Dataset
from hub.core.tests.common import parametrize_all_dataset_storages


@parametrize_all_dataset_storages
def test_delete_with_single_tensor(ds: Dataset):
    tensor = ds.create_tensor("tensor")

    tensor.append(np.ones((100, 100)))
    tensor.extend(np.ones((10, 100, 100)))

    assert len(ds) == 11
    assert ds.tensor_names == ("tensor",)

    ds.delete_tensor("tensor")

    _assert_dead_tensor(tensor)

    assert len(ds) == 0
    assert ds.tensor_names == tuple()

    _assert_tensor_deleted_from_core("tensor", ds.storage)

    recreated_tensor = ds.create_tensor("tensor")

    recreated_tensor.append(np.zeros((5, 5)))

    assert len(recreated_tensor) == 1
    assert len(ds) == 1
    np.testing.assert_array_equal(recreated_tensor.numpy(), np.zeros((1, 5, 5)))


@parametrize_all_dataset_storages
def test_delete_with_multiple_tensors(ds: Dataset):
    tensor = ds.create_tensor("tensor")
    other_tensor = ds.create_tensor("other_tensor")

    tensor.extend(np.ones((10, 100, 100)))
    other_tensor.extend(np.ones((9, 100, 100)))

    assert len(ds) == 9
    assert ds.tensor_names == ("tensor", "other_tensor")

    ds.delete_tensor("other_tensor")

    _assert_dead_tensor(other_tensor)

    assert ds.tensor_names == ("tensor",)
    assert len(ds) == 10
    np.testing.assert_array_equal(tensor.numpy(), np.ones((10, 100, 100)))

    _assert_tensor_deleted_from_core("other_tensor", ds.storage)

    recreated_other_tensor = ds.create_tensor("other_tensor")

    recreated_other_tensor.append(np.zeros((5, 5)))

    assert len(recreated_other_tensor) == 1
    assert len(ds) == 1
    np.testing.assert_array_equal(recreated_other_tensor.numpy(), np.zeros((1, 5, 5)))


def _assert_tensor_deleted_from_core(name: str, storage: StorageProvider):
    # read directly from core to make sure it's gone

    # cannot use `tensor_exists` here because that checks if both `index_meta` + `tensor_meta` are there, but we need to check them individually
    tensor_meta_key = get_tensor_meta_key(name)
    index_meta_key = get_index_meta_key(name)
    assert tensor_meta_key not in storage
    assert index_meta_key not in storage

    with pytest.raises(TensorDoesNotExistError):
        read_samples_from_tensor(name, storage)


def _assert_dead_tensor(tensor: Tensor):
    with pytest.raises(TensorDoesNotExistError):
        tensor.append(np.ones((10, 10)))
    with pytest.raises(TensorDoesNotExistError):
        tensor.extend(np.ones((10, 10)))
    with pytest.raises(TensorDoesNotExistError):
        tensor.numpy()
