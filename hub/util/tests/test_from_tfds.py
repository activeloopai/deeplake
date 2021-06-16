import numpy as np
import pytest

from hub.core.tests.common import parametrize_all_dataset_storages
from hub.util.check_installation import tfds_installed, requires_tfds, requires_tensorflow  # type: ignore


@requires_tfds
def test_from_tfds_to_path(local_storage):
    if local_storage is None:
        pytest.skip()
    from hub.util.from_tfds import from_tfds, from_tfds_to_path  # type: ignore

    if local_storage is None:
        pytest.skip()
    hub_ds = from_tfds_to_path(
        tfds_dataset_name="mnist",
        split="test",
        hub_ds_path=local_storage.root,
        batch_size=100,
    )
    assert hub_ds.image.shape == (10000, 28, 28, 1)


@requires_tensorflow
@requires_tfds
@parametrize_all_dataset_storages
def test_from_tfds(ds):
    import tensorflow_datasets as tfds  # type: ignore
    from hub.util.from_tfds import from_tfds, from_tfds_to_path  # type: ignore

    tfds_ds = tfds.load("mnist", split="train").batch(1).take(10)
    from_tfds(tfds_ds=tfds_ds, ds=ds)
    for i, example in enumerate(tfds_ds):
        image, label = example["image"], example["label"]
        img = image[0, :, :, :].numpy()
        ds_img = ds.image[i].numpy()
        np.testing.assert_array_equal(img, ds_img)
