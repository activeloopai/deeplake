import deeplake
from deeplake.tests.common import requires_libdeeplake
from deeplake.core.dataset.deeplake_query_dataset import DeepLakeQueryDataset
import pytest
import numpy as np


@requires_libdeeplake
def test_single_source_query():
    ds = deeplake.query('SELECT * FROM "hub://activeloop/mnist-train"')
    assert len(ds) == 60000
    assert len(ds.tensors) == 2
    assert ds.images.meta.htype == "image"
    assert ds.labels.meta.htype == "class_label"

    ds = deeplake.query('SELECT images FROM "hub://activeloop/mnist-train"')
    assert len(ds) == 60000
    assert len(ds.tensors) == 1
    assert ds.images.meta.htype == "image"


@requires_libdeeplake
def test_multi_source_query():
    with pytest.raises(RuntimeError):
        ds = deeplake.query(
            'SELECT * FROM "hub://activeloop/mnist-train" UNION (SELECT * FROM "hub://activeloop/imagenet-train")'
        )

    ds = deeplake.query(
        'SELECT * FROM "hub://activeloop/mnist-train" UNION (SELECT images, labels FROM "hub://activeloop/imagenet-train")'
    )
    assert len(ds) == 1341166
    assert len(ds.tensors) == 2
    assert ds.images.meta.htype == "image"
    assert ds.labels.meta.htype == "class_label"

    ds = deeplake.query(
        'SELECT * FROM (SELECT * FROM "hub://activeloop/mnist-train" UNION (SELECT images, labels FROM "hub://activeloop/imagenet-train")) WHERE labels == 0'
    )
    assert len(ds) == 7223
    assert len(ds.tensors) == 2
    d = ds.labels.numpy()
    assert np.all(d == 0)
