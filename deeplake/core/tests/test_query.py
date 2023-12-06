import deeplake
from deeplake.tests.common import requires_libdeeplake
from deeplake.core.dataset.deeplake_query_dataset import DeepLakeQueryDataset
import pytest


@requires_libdeeplake
def test_single_sourcequery():
    ds = deeplake.query('SELECT * FROM "hub://activeloop/mnist-train"')
    assert len(ds) == 60000
    assert len(ds.tensors) == 2
    assert ds.images.meta.htype == "image"
    assert ds.labels.meta.htype == "class_label"

    ds = deeplake.query('SELECT images FROM "hub://activeloop/mnist-train"')
    assert len(ds) == 60000
    assert len(ds.tensors) == 1
    assert ds.images.meta.htype == "image"
