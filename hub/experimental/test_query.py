from math import floor
from hub.tests.common import requires_linux
from hub.experimental import query


@requires_linux
def test_query(local_ds):
    with local_ds as ds:
        ds.create_tensor("label")
        for i in range(100):
            ds.label.append(floor(i / 20))

    dsv = query(local_ds, "SELECT * WHERE CONTAINS(label, 2)")
    assert len(dsv) == 20
    for i in range(20):
        assert dsv.label[i].numpy() == 2
