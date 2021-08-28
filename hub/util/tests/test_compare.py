from hub.util import compare
from hub.api.dataset import dataset
from hub.core.dataset import Dataset
from hub.util.exceptions import (
    TensorAlreadyLinkedError,
    TensorAlreadyExistsError,
    InvalidTensorNameError,
    TensorDoesNotExistError,
    LinkedTensorError,
    HashesTensorDoesNotExistError,
)
from hub.tests.common import get_dummy_data_path
import hub, pytest
import numpy as np
from hub.core.tensor import Tensor, add_missing_meta_attributes
from hub.tests.common import TENSOR_KEY
from hub.constants import HASHES_TENSOR_FOLDER
from hub.tests.dataset_fixtures import enabled_datasets

import glob
import os


def test_compare_np_arrays(memory_ds, memory_ds_2):

    with memory_ds:
        memory_ds.create_tensor("image", hash_samples=True)
        for i in range(10):
            memory_ds.image.append(np.ones((28, 28), dtype=np.uint8) * i)

    with memory_ds_2:
        memory_ds_2.create_tensor("image", hash_samples=True)
        for i in range(10):
            memory_ds_2.image.append(np.ones((28, 28), dtype=np.uint8) * i)

    assert hub.compare(memory_ds, memory_ds_2) == 1.0


def test_compare_half_np_arrays(memory_ds, memory_ds_2):

    with memory_ds:
        memory_ds.create_tensor("image", hash_samples=True)
        for i in range(5):
            memory_ds.image.append(np.ones((28, 28), dtype=np.uint8) * i)

    with memory_ds_2:
        memory_ds_2.create_tensor("image", hash_samples=True)
        for i in range(10):
            memory_ds_2.image.append(np.ones((28, 28), dtype=np.uint8) * i)

    assert hub.compare(memory_ds, memory_ds_2) == 0.5


def test_compare_image_datasets(memory_ds, memory_ds_2):

    dataset_1 = glob.glob(get_dummy_data_path("tests_compare/dataset_1/*"))
    dataset_2 = glob.glob(get_dummy_data_path("tests_compare/dataset_2/*"))

    with memory_ds:
        memory_ds.create_tensor(
            "images", htype="image", sample_compression="jpeg", hash_samples=True
        )

        for label, folder_path in enumerate(dataset_1):
            paths = glob.glob(os.path.join(folder_path, "*"))  # Get subfolders

            # Iterate through images in the subfolders
            for path in paths:
                memory_ds.images.append(
                    hub.read(path)
                )  # Append to images tensor using hub.read

    with memory_ds_2:
        memory_ds_2.create_tensor(
            "images", htype="image", sample_compression="jpeg", hash_samples=True
        )

        for label, folder_path in enumerate(dataset_2):
            paths = glob.glob(os.path.join(folder_path, "*"))  # Get subfolders

            # Iterate through images in the subfolders
            for path in paths:
                memory_ds_2.images.append(
                    hub.read(path)
                )  # Append to images tensor using hub.read

    assert memory_ds.hidden_tensors[HASHES_TENSOR_FOLDER].meta.is_linked_tensor == True
    assert HASHES_TENSOR_FOLDER in memory_ds.images.meta.linked_tensors
    assert hub.compare(memory_ds, memory_ds_2) == 0.5


def test_compare_errors(memory_ds, memory_ds_2):
    memory_ds.create_tensor("image")
    memory_ds_2.create_tensor("image")

    with pytest.raises(HashesTensorDoesNotExistError):
        hub.compare(memory_ds, memory_ds_2)


@enabled_datasets
def test_linked_tensors(ds):

    ds.create_tensor("image")
    ds.create_tensor("grayscale_image")
    ds.create_tensor("cropped_image")

    ds._link_tensor(ds.image, ds.grayscale_image)
    ds._link_tensor(ds.image, ds.cropped_image)

    assert ds.grayscale_image.meta.is_linked_tensor == True
    assert "grayscale_image" in ds.image.meta.linked_tensors
    assert "cropped_image" in ds.image.meta.linked_tensors

    with pytest.raises(TensorAlreadyExistsError):
        ds._create_hidden_tensor("hidden_tensor")
        ds._create_hidden_tensor("hidden_tensor")

    with pytest.raises(LinkedTensorError):
        ds.grayscale_image.append([1, 2, 3])

    with pytest.raises(LinkedTensorError):
        ds.grayscale_image[0] = np.zeros((3, 3))

    with pytest.raises(InvalidTensorNameError):
        ds._create_hidden_tensor("tensors")

    with pytest.raises(TensorAlreadyLinkedError):
        ds.create_tensor("rotated_image")
        ds._link_tensor(ds.grayscale_image, ds.rotated_image)


def test_missing_meta_attributes():

    ds = hub.dataset(get_dummy_data_path("tests_compare/dataset_with_old_meta/"))

    add_missing_meta_attributes(
        ds["images"].key, ds["images"].storage, ds["images"].meta
    )

    assert ds.images.meta.linked_tensors == []
    assert ds.images.meta.is_linked_tensor == False
    assert ds.images.meta.hash_samples == False

    with pytest.raises(TensorDoesNotExistError):
        add_missing_meta_attributes(
            ds["no_image"].key, ds["no_image"].storage, ds["no_image"].meta
        )
