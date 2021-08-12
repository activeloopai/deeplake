from hub.util import compare
from hub.api.dataset import dataset
from hub.core.dataset import Dataset
import hub, pytest
import numpy as np
from PIL import Image
from hub.core.tensor import Tensor
from hub.tests.common import TENSOR_KEY

def test_compare_np_arrays():

    path_1 = "test_dataset_load_1"
    path_2 = "test_dataset_load_2"

    ds_1 = hub.empty(path_1)
    ds_2 = hub.empty(path_2)
    
    with ds_1:
        ds_1.create_tensor("image", hash_samples=True)
        for i in range(10):
            ds_1.image.append(np.ones((28, 28), dtype=np.uint8) * i)

    with ds_2:
        ds_2.create_tensor("image", hash_samples=True)
        for i in range(10):
            ds_2.image.append(np.ones((28, 28), dtype=np.uint8) * i)

    assert hub.compare(ds_1, ds_2) == 1.0
    
    ds_1.delete()
    ds_2.delete()

def test_compare_half_np_arrays():

    path_1 = "test_dataset_load_1"
    path_2 = "test_dataset_load_2"

    ds_1 = hub.empty(path_1)
    ds_2 = hub.empty(path_2)
    
    with ds_1:
        ds_1.create_tensor("image", hash_samples=True)
        for i in range(5):
            ds_1.image.append(np.ones((28, 28), dtype=np.uint8) * i)

    with ds_2:
        ds_2.create_tensor("image", hash_samples=True)
        for i in range(10):
            ds_2.image.append(np.ones((28, 28), dtype=np.uint8) * i)

    assert hub.compare(ds_1, ds_2) == 0.5
    
    ds_1.delete()
    ds_2.delete()
