import numpy as np
import pytest
import tensorflow_datasets as tfds

from core.tests.common import parametrize_all_dataset_storages
from tools.from_tfds import from_tfds, from_tfds_to_path


@pytest.mark.full_benchmark
def test_from_tfds_to_path(local_storage):
    if local_storage is None:
        pytest.skip()
    hub_ds = from_tfds_to_path(
        tfds_dataset_name="mnist",
        split="test",
        hub_ds_path=local_storage.root,
        batch_size=100,
    )
    assert len(hub_ds) == 10000
    assert hub_ds.image.shape.upper == (28, 28)
    assert hub_ds.image.shape.lower == (28, 28)
    assert hub_ds.image[1000].shape.upper == (28, 28)
    assert hub_ds.image[1000].shape.lower == (28, 28)


@parametrize_all_dataset_storages
def test_from_tfds(ds):
    tfds_ds = tfds.load("mnist", split="train").batch(1).take(10)
    from_tfds(tfds_ds=tfds_ds, ds=ds)
    for i, example in enumerate(tfds_ds):
        image, label = example["image"], example["label"]
        img = image[0, :, :, 0].numpy()
        ds_img = ds.image[i].numpy()
        np.testing.assert_array_equal(img, ds_img)
