import numpy as np
import tensorflow_datasets as tfds

from core.tests.common import parametrize_all_dataset_storages
from tools.from_tfds import from_tfds


@parametrize_all_dataset_storages
def test_from_tfds(ds):
    tfds_ds = tfds.load("mnist", split="train").batch(10).take(100)
    from_tfds(tfds_ds=tfds_ds, ds=ds)
    for i, example in enumerate(tfds_ds):
        image, label = example["image"], example["label"]
        img = image.numpy()
        ds_img = ds.image[i].numpy()
        np.testing.assert_array_equal(img, ds_img)

