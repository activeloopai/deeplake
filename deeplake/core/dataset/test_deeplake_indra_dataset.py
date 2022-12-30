import deeplake as dp
import numpy as np
import pytest
from deeplake.core.dataset.deeplake_indra_dataset import DeepLakeIndraDataset
from indra import api
import functools


# all of these tests should be performed for hub, s3 and local storage

path = "hub://activeloop-test/mnist-train"
deeplake_ds = dp.load(path)
indra_ds = api.dataset(path)
indra_ds = ds_indra[:100]
deeplake_indra_ds = DeepLakeIndraDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)


def test_indexing(deeplake_indra_ds, deeplake_ds, indra_ds):

    assert len(deeplake_indra_ds) == len(indra_ds)
    assert deeplake_indra_ds.__getstate__() == deeplake_ds.__getstate__()

    np.isclose(deeplake_indra_ds.images.numpy(), indra_ds.images.numpy())
    np.isclose(deeplake_indra_ds.images[5:55].numpy(), indra_ds.images[5:55].numpy())
    np.isclose(deeplake_indra_ds[5:55].images.numpy(), indra_ds[5:55].images.numpy())


def test_save_view(deeplake_indra_ds, deeplake_ds, indra_ds):
    deeplake_indra_ds.save_view()
    assert (
        deeplake_indra_ds.base_storage["queries.json"]
        == deeplake_ds.base_storage["queries.json"]
    )


def test_load_view(deeplake_indra_ds, deeplake_ds, indra_ds, view_id):
    deeplake_ds.load_view(view_id)
    assert np.isclose(indra_ds.tensors[0][:], deeplake_indra_ds.image.numpy())
    # can it be that i load view after run group by query?
    deeplake_indra_ds.load_view(view_id)
    assert np.isclose(indra_ds.tensors[0][:], deeplake_indra_ds.image.numpy())


def test_query():
    # can we run queries sequentially?
    pass


def test_copy():
    pass


def test_optimize_views():
    pass


def test_parallel_computing():
    pass


def test_acessing_data(deeplake_indra_ds, deeplake_ds, indra_ds):
    assert np.isclose(
        deeplake_indra_ds.images.numpy(), deeplake_indra_ds["images"].numpy()
    )

    # test with hirarcy
    # some code goes here

    deeplake_indra_ds_data = deeplake_indra_ds.labels[0].data()
    deeplake_indra_ds_value = deeplake_indra_ds_data["value"]
    assert isinstance(deeplake_indra_ds_value, dict)
    assert len(deeplake_indra_ds_value) == 100
    assert isinstance(deeplake_indra_ds_value, list)
