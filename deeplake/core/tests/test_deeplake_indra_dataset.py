import deeplake as dp
import numpy as np
import pytest
from deeplake.core.dataset.dataset import DeepLakeQueryDataset
from indra import api
import functools
import os


# all of these tests should be performed for hub, s3 and local storage

home_dir = "/Users/adilkhansarsen/Documents/work/Hub_2/"
path = "/Users/adilkhansarsen/Documents/work/Hub_2/deeplake/mnist-train"
TOKEN = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTY2MDI5MzExNiwiZXhwIjo0ODEzODkzMTE2fQ.eyJpZCI6ImFkaWxraGFuIn0.wyvhu0z1ak72eyeiq8VKYKT9R268D0i4jH8724X6G0BJLFQNKFHL_BkD5BVDtPln3AdGkNHKeCM8Og2DQ038gA"

indra_ds = api.dataset(path, token=TOKEN)
indra_ds = indra_ds[:100]


def test_indexing():
    deeplake_ds = dp.load(path, token=TOKEN)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)

    assert len(deeplake_indra_ds) == len(indra_ds)
    assert deeplake_indra_ds.__getstate__() == deeplake_ds.__getstate__()

    np.isclose(
        deeplake_indra_ds.images.numpy(),
        np.dstack(indra_ds.tensors[0][:]).transpose(2, 0, 1),
    )
    np.isclose(
        deeplake_indra_ds.images[5:55].numpy(),
        np.dstack(indra_ds.tensors[0][5:55]).transpose(2, 0, 1),
    )

    np.isclose(
        deeplake_indra_ds[5:55].images.numpy(),
        np.dstack(indra_ds.tensors[0][5:55]).transpose(2, 0, 1),
    )


def test_save_view():
    deeplake_ds = dp.load(path, token=TOKEN)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)
    deeplake_indra_ds.save_view()
    assert (
        deeplake_indra_ds.base_storage["queries.json"]
        == deeplake_ds.base_storage["queries.json"]
    )


def test_load_view():
    deeplake_ds = dp.load(path, token=TOKEN)
    query_str = "SELECT * GROUP BY labels"
    view = deeplake_ds.query(query_str)
    view_path = view.save_view()
    view_id = "96c542aca468cb8261356b400dc9f9a753f0b43432c75337bdee10a91ce5702a"
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)
    view = deeplake_ds.load_view(view_id)
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


def test_acessing_data(path, indra_ds):
    deeplake_ds = dp.load(path, token=TOKEN)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)

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
