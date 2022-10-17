from deeplake.core.multi_view import MultiDatasetView

import numpy as np

import deeplake
import pytest


def create_ragged_ds(ds):
    with ds:
        ds.create_tensor("images", htype="image", sample_compression="jpg")
        ds.create_tensor("labels", htype="class_label")

        ds.images.extend(np.random.randint(0, 255, (10, 5, 5, 3), dtype=np.uint8))
        ds.labels.extend(np.zeros((8,), dtype=np.uint32))

    return ds


def create_good_ds(ds):
    with ds:
        ds.create_tensor("images", htype="image", sample_compression="jpg")
        ds.create_tensor("labels", htype="class_label")

        ds.images.extend(np.random.randint(0, 255, (10, 5, 5, 3), dtype=np.uint8))
        ds.labels.extend(np.zeros((10,), dtype=np.uint32))

    return ds


def test_ds_compatibilty(local_path):
    ds = deeplake.empty(local_path, overwrite=True)
    ds = create_good_ds(ds)

    ds2 = deeplake.empty(f"{local_path}_2", overwrite=True)

    assert not MultiDatasetView.is_compatible(ds, ds2)

    ds2.create_tensor("images", htype="image", sample_compression="png")
    ds2.create_tensor("labels", htype="class_label")

    assert not MultiDatasetView.is_compatible(ds, ds2)

    ds2 = deeplake.empty(ds2.path, overwrite=True)
    ds2.create_tensor("images", htype="image", sample_compression="jpg")
    ds2.create_tensor("labels", htype="class_label")

    assert MultiDatasetView.is_compatible(ds, ds2)

    ds.delete()
    ds2.delete()


def test_dataset_add(local_path):
    ds1 = deeplake.empty(local_path, overwrite=True)
    ds2 = deeplake.empty(f"{local_path}_2", overwrite=True)

    ds1 = create_good_ds(ds1)
    ds2 = create_good_ds(ds2)

    multiview = ds1 + ds2

    assert len(multiview) == 20

    # single index
    for i, sample in enumerate(ds1):
        np.testing.assert_array_equal(
            multiview[i].images.numpy(), sample.images.numpy()
        )
        np.testing.assert_array_equal(
            multiview.images[i].numpy(), sample.images.numpy()
        )

    len_ds2 = len(ds2)
    for i, sample in enumerate(ds2):
        np.testing.assert_array_equal(
            multiview[len_ds2 + i].images.numpy(), sample.images.numpy()
        )
        np.testing.assert_array_equal(
            multiview.images[len_ds2 + i].numpy(), sample.images.numpy()
        )

    # slice indexing
    np.testing.assert_array_equal(
        multiview[5:15:3].images.numpy(),
        np.vstack([ds1[5::3].images.numpy(), ds2[1:5:3].images.numpy()]),
    )

    np.testing.assert_array_equal(
        multiview[-2:2:-4].images.numpy(),
        np.vstack([ds2[8::-4].images.numpy(), ds1[6:2:-4].images.numpy()]),
    )

    # list indexing
    np.testing.assert_array_equal(
        multiview[[0, 1, 2, 3, 4]].images.numpy(), ds1[:5].images.numpy()
    )
    np.testing.assert_array_equal(
        multiview[[8, 9, 10, 11, 12]].images.numpy(),
        np.vstack([ds1[[8, 9]].images.numpy(), ds2[[0, 1, 2]].images.numpy()]),
    )

    # partial samples
    np.testing.assert_array_equal(
        multiview[:10, :2, :2].images.numpy(), ds1[:10, :2, :2].images.numpy()
    )
    np.testing.assert_array_equal(
        multiview.images[8:12, :2, :2].numpy(),
        np.vstack([ds1[8:, :2, :2].images.numpy(), ds2[:2, :2, :2].images.numpy()]),
    )

    ds1 = deeplake.empty(local_path, overwrite=True)
    ds2 = deeplake.empty(f"{local_path}_2", overwrite=True)

    ds1 = create_ragged_ds(ds1)
    ds2 = create_good_ds(ds2)

    multiview = ds1 + ds2

    assert len(multiview) == 20

    assert len(multiview.images) == 20
    assert len(multiview.labels) == 18

    np.testing.assert_array_equal(
        multiview.images[8:12, :2, :2].numpy(),
        np.vstack([ds1[8:, :2, :2].images.numpy(), ds2[:2, :2, :2].images.numpy()]),
    )
    np.testing.assert_array_equal(
        multiview.labels[5::2].numpy(),
        np.vstack([ds1.labels[5::2].numpy(), ds2[::2].labels.numpy()]),
    )

    ds1.delete()
    ds2.delete()
