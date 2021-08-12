from hub.util import compare
from hub.api.dataset import dataset
from hub.core.dataset import Dataset
from hub.tests.common import get_dummy_data_path
import hub, pytest
import numpy as np
from PIL import Image
from hub.core.tensor import Tensor
from hub.tests.common import TENSOR_KEY

import glob
import os


def test_compare_np_arrays():

    path_1 = "np_dataset_1"
    path_2 = "np_dataset_2"

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

    path_1 = "np_half_dataset_1"
    path_2 = "np_half_dataset_2"

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

def test_compare_image_datasets():
    dataset_1 = glob.glob(get_dummy_data_path("tests_compare/dataset_1/*"))
    dataset_2 = glob.glob(get_dummy_data_path("tests_compare/dataset_2/*"))
    
    hub_path_1 = "test_dataset_1"
    hub_path_2 = "test_dataset_2"

    ds_1 = hub.dataset(hub_path_1)
    ds_2 = hub.dataset(hub_path_2)
    
    with ds_1:
        ds_1.create_tensor('images', htype = 'image', sample_compression='jpeg', hash_samples=True)

        for label, folder_path in enumerate(dataset_1):
            paths = glob.glob(os.path.join(folder_path, '*')) # Get subfolders
        
            # Iterate through images in the subfolders
            for path in paths:
                ds_1.images.append(hub.read(path))  # Append to images tensor using hub.read

    with ds_2:
        ds_2.create_tensor('images', htype = 'image', sample_compression='jpeg', hash_samples=True)
    
        for label, folder_path in enumerate(dataset_2):
            paths = glob.glob(os.path.join(folder_path, '*')) # Get subfolders
        
            # Iterate through images in the subfolders
            for path in paths:
                ds_2.images.append(hub.read(path))  # Append to images tensor using hub.read
    
    
    assert hub.compare(ds_1, ds_2) == 0.5
    
    ds_1.delete()
    ds_2.delete()