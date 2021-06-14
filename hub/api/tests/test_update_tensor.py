from hub.api.dataset import Dataset
import numpy as np
from hub.core.tests.common import parametrize_all_dataset_storages


@parametrize_all_dataset_storages
def test_update_fixed_tensor(ds: Dataset):
    image = ds.create_tensor("image")
    image.extend(np.ones((20, 28, 28, 3)))

    assert image.shape.lower == (28, 28, 3)
    assert image.shape.upper == (28, 28, 3)

    # fixed samples
    image[5] = np.arange(28 * 28 * 3).reshape((28, 28, 3))

    assert len(image) == 20
    assert image.shape.lower == (28, 28, 3)
    assert image.shape.upper == (28, 28, 3)

    assert image.numpy() == np.ones((20, 28, 28, 3))


@parametrize_all_dataset_storages
def test_update_dynamic_tensor(ds: Dataset):
    image = ds.create_tensor("image")
    image.extend(np.ones((20, 28, 28, 3)))

    assert image.shape.lower == (28, 28, 3)
    assert image.shape.upper == (28, 28, 3)

    # dynamic samples
    image[5] = np.arange(10 * 30 * 3).reshape((10, 30, 3))

    assert len(image) == 20
    assert image.shape.lower == (10, 28, 3)
    assert image.shape.upper == (28, 30, 3)

    assert image.numpy(aslist=True) == [
        *np.ones((5, 28, 28, 3)),
        np.arange(10 * 30 * 3).reshape((10, 30, 3)),
        *np.ones((14, 28, 28, 3)),
    ]


@parametrize_all_dataset_storages
def test_update_tensor_slice(ds: Dataset):
    image = ds.create_tensor("image")
    image.extend(np.ones((20, 28, 28, 3)))

    assert image.shape.lower == (28, 28, 3)
    assert image.shape.upper == (28, 28, 3)

    # fixed samples
    image[5:10] = np.arange(5 * 10 * 30 * 3).reshape((5, 10, 30, 3))

    assert len(image) == 20
    assert image.shape.lower == (10, 28, 3)
    assert image.shape.upper == (28, 30, 3)

    assert image.numpy(aslist=True) == [
        *np.ones((5, 28, 28, 3)),
        *np.arange(5 * 10 * 30 * 3).reshape((5, 10, 30, 3)),
        *np.ones((10, 28, 28, 3)),
    ]


@parametrize_all_dataset_storages
def test_update_tensor_slice_with_list(ds: Dataset):
    image = ds.create_tensor("image")
    image.extend(np.ones((20, 28, 28, 3)))

    assert image.shape.lower == (28, 28, 3)
    assert image.shape.upper == (28, 28, 3)

    # dynamic samples
    image[5:10] = [
        *np.arange(3 * 10 * 30 * 3).reshape((3, 10, 30, 5)),
        *np.arange(2 * 90 * 3 * 2).reshape((2, 90, 3, 2)),
    ]

    assert len(image) == 20
    assert image.shape.lower == (10, 3, 2)
    assert image.shape.upper == (90, 30, 5)

    assert image.numpy(aslist=True) == [
        *np.ones((5, 28, 28, 3)),
        *np.arange(3 * 10 * 30 * 5).reshape((3, 10, 30, 5))
        * np.arange(2 * 90 * 3 * 2).reshape((2, 90, 3, 2))
        * np.ones((14, 28, 28, 3)),
    ]
