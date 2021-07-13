import numpy as np

from hub.util.check_installation import requires_tfds, requires_tensorflow  # type: ignore
from hub.tests.dataset_fixtures import enabled_datasets


@requires_tfds
def test_from_tfds_to_path(local_storage):
    from hub.util.from_tfds import from_tfds_to_path  # type: ignore

    hub_ds = from_tfds_to_path(
        tfds_dataset_name="mnist",
        split="test",
        hub_ds_path=local_storage.root,
        batch_size=100,
    )
    assert hub_ds.image.shape == (10000, 28, 28, 1)


@requires_tensorflow
@requires_tfds
@enabled_datasets
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
