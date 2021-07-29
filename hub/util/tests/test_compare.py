from hub.util import compare
from hub.api.dataset import dataset
from hub.core.dataset import Dataset
import hub, pytest
import numpy as np
from PIL import Image
from hub.util.exceptions import HashlistDoesNotExistError


def test_compare_tensors(memory_ds):

    ds = memory_ds

    ds.create_tensor("ints1", dtype="int64", isHash=True)
    ds.ints1.extend(np.arange(10, dtype="int64"))

    ds.create_tensor("ints2", dtype="int64", isHash=True)
    ds.ints2.extend(np.arange(10, dtype="int64"))

    # Jaccard similarity score should be 1.0 as both hashlists are same
    assert hub.compare(ds.ints1, ds.ints2) == 1.0


def test_compare_half_tensors(memory_ds):

    ds = memory_ds

    ds.create_tensor("ints1", dtype="int64", isHash=True)
    ds.ints1.extend(np.arange(10, dtype="int64"))

    ds.create_tensor("ints2", dtype="int64", isHash=True)
    ds.ints2.extend(np.arange(5, dtype="int64"))

    # Jaccard similarity score should be 0.5 in this case
    assert hub.compare(ds.ints1, ds.ints2) == 0.5


def test_hashlist_does_not_exist(memory_ds):

    ds = memory_ds

    ds.create_tensor("ints1", dtype="int64")
    ds.ints1.extend(np.arange(10, dtype="int64"))

    ds.create_tensor("ints2", dtype="int64")
    ds.ints2.extend(np.arange(10, dtype="int64"))

    with pytest.raises(HashlistDoesNotExistError):
        hub.compare(ds.ints1, ds.ints2)
