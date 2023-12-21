import deeplake
from deeplake.tests.common import requires_libdeeplake
from deeplake.core.dataset.deeplake_query_dataset import DeepLakeQueryDataset
from deeplake.client.client import DeepLakeBackendClient
import pytest
import numpy as np


@requires_libdeeplake
def test_single_source_query(
    hub_cloud_dev_credentials,
):
    username, password = hub_cloud_dev_credentials
    token = DeepLakeBackendClient().request_auth_token(username, password)
    ds = deeplake.query('SELECT * FROM "hub://activeloop/mnist-train"', token=token)
    assert len(ds) == 60000
    assert len(ds.tensors) == 2
    assert ds.images.meta.htype == "image"
    assert ds.labels.meta.htype == "class_label"

    ds = deeplake.query(
        'SELECT images FROM "hub://activeloop/mnist-train"', token=token
    )
    assert len(ds) == 60000
    assert len(ds.tensors) == 1
    assert ds.images.meta.htype == "image"


@requires_libdeeplake
def test_multi_source_query(hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials
    token = DeepLakeBackendClient().request_auth_token(username, password)

    with pytest.raises(RuntimeError):
        ds = deeplake.query(
            'SELECT * FROM "hub://activeloop/mnist-train" UNION (SELECT * FROM "hub://activeloop/coco-train")',
            token=token,
        )

    ds = deeplake.query(
        'SELECT * FROM "hub://activeloop/mnist-train" UNION (SELECT images, categories[0] as labels FROM "hub://activeloop/coco-train")',
        token=token,
    )
    assert len(ds) == 178287
    assert len(ds.tensors) == 2
    assert ds.images.meta.htype == "image"
    assert ds.labels.meta.htype == "class_label"

    ds = deeplake.query(
        'SELECT * FROM (SELECT * FROM "hub://activeloop/mnist-train" UNION (SELECT images, labels FROM "hub://activeloop/cifar100-train")) WHERE labels == 0',
        token=token,
    )
    assert len(ds) == 6423
    assert len(ds.tensors) == 2
    d = ds.labels.numpy()
    assert np.all(d == 0)
