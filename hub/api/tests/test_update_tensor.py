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

    # TODO: what do we do with `tensor`? the reference should die?
    print(tensor)

    assert len(ds) == 0
    assert ds.tensor_names == tuple()

    _assert_tensor_deleted_from_core("tensor", ds.storage)


@parametrize_all_dataset_storages
def test_delete_with_multiple_tensors(ds: Dataset):
    tensor = ds.create_tensor("tensor")
    other_tensor = ds.create_tensor("other_tensor")

    tensor.extend(np.ones((10, 100, 100)))
    other_tensor.extend(np.ones((9, 100, 100)))

    assert len(ds) == 9
    assert ds.tensor_names == ("tensor", "other_tensor")

    ds.delete_tensor("other_tensor")

    # TODO: what do we do with `other_tensor`? the reference should die?
    assert len(ds) == 10
    assert ds.tensor_names == ("tensor",)

    _assert_tensor_deleted_from_core("other_tensor", ds.storage)


def _assert_tensor_deleted_from_core(name: str, storage: StorageProvider):
    # read directly from core to make sure it's gone
    assert not tensor_exists("tensor", storage)
    with pytest.raises(TensorDoesNotExistError):
        read_samples_from_tensor("tensor", storage)
