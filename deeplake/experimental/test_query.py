from math import floor
from deeplake.tests.common import requires_libdeeplake


@requires_libdeeplake
def test_query(local_ds):
    with local_ds as ds:
        ds.create_tensor("label")
        for i in range(100):
            ds.label.append(floor(i / 20))

    dsv = local_ds.query("SELECT * WHERE CONTAINS(label, 2)")
    assert len(dsv) == 20
    for i in range(20):
        assert dsv.label[i].numpy() == 2


@requires_libdeeplake
def test_sample(local_ds):
    with local_ds as ds:
        ds.create_tensor("label")
        for i in range(100):
            ds.label.append(floor(i / 20))

    dsv = local_ds.sampler(
        "max_weight(label == 2: 10, label == 1: 1)", replace=False, size=10
    )
    assert len(dsv) == 10
    for i in range(10):
        assert dsv.label[i].numpy() == 2 or dsv.label[i].numpy() == 1

    dsv = local_ds.sampler("max_weight(label == 2: 10, label == 1: 1)", replace=True)
    assert len(dsv) == 100
    for i in range(100):
        assert dsv.label[i].numpy() == 2 or dsv.label[i].numpy() == 1

    dsv = local_ds.sampler("label")
    assert len(dsv) == 100

    weights = list()
    for i in range(100):
        weights.append(1 if floor(i / 20) == 0 else 0)

    dsv = local_ds.sampler(weights)
    assert len(dsv) == 100
    for i in range(100):
        assert dsv.label[i].numpy() == 0
