from math import floor
from deeplake.tests.common import requires_libdeeplake
import numpy as np


@requires_libdeeplake
def test_query(hub_cloud_ds):
    with hub_cloud_ds as ds:
        ds.create_tensor("label")
        for i in range(100):
            ds.label.append(floor(i / 20))

    dsv = hub_cloud_ds.query("SELECT * WHERE CONTAINS(label, 2)")
    assert len(dsv) == 20
    for i in range(20):
        assert dsv.label[i].numpy() == 2


@requires_libdeeplake
def test_sample(hub_cloud_ds):
    with hub_cloud_ds as ds:
        ds.create_tensor("label")
        for i in range(100):
            ds.label.append(floor(i / 20))

    dsv = hub_cloud_ds.sample_by(
        "max_weight(label == 2: 10, label == 1: 1)", replace=False, size=10
    )
    assert len(dsv) == 10
    for i in range(10):
        assert dsv.label[i].numpy() == 2 or dsv.label[i].numpy() == 1

    dsv = hub_cloud_ds.sample_by(
        "max_weight(label == 2: 10, label == 1: 1)", replace=True
    )
    assert len(dsv) == 100
    for i in range(100):
        assert dsv.label[i].numpy() == 2 or dsv.label[i].numpy() == 1

    dsv = hub_cloud_ds.sample_by("label")
    assert len(dsv) == 100

    weights = list()
    for i in range(100):
        weights.append(1 if floor(i / 20) == 0 else 0)

    dsv = hub_cloud_ds.sample_by(weights)
    assert len(dsv) == 100
    for i in range(100):
        assert dsv.label[i].numpy() == 0

    weights = np.ndarray((100), np.int32)
    for i in range(100):
        weights[i] = 1 if floor(i / 10) == 0 else 0

    dsv = hub_cloud_ds.sample_by(weights)
    assert len(dsv) == 100
    for i in range(100):
        assert dsv.label[i].numpy() == 0
