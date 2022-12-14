import pytest

import deeplake
from deeplake.util.exceptions import InvalidSourcePathError


def test_connect_dataset_api(
    hub_cloud_dev_token,
    hub_cloud_path,
    hub_cloud_dev_managed_creds_key,
    s3_ds_generator,
):
    s3_ds = s3_ds_generator()
    s3_ds.create_tensor("x")
    s3_ds.x.append(10)

    ds = deeplake.connect(
        s3_ds.path,
        creds_key=hub_cloud_dev_managed_creds_key,
        dest_path=hub_cloud_path,
        token=hub_cloud_dev_token,
    )

    assert ds is not None
    assert ds.path.startswith("hub://")
    assert "x" in ds.tensors
    assert ds.x[0].numpy() == 10


def test_in_place_dataset_connect(
    hub_cloud_dev_token,
    hub_cloud_path,
    hub_cloud_dev_managed_creds_key,
    s3_ds_generator,
):
    s3_ds = s3_ds_generator()
    s3_ds.create_tensor("x")
    s3_ds.x.append(10)

    s3_ds.connect(
        creds_key=hub_cloud_dev_managed_creds_key,
        dest_path=hub_cloud_path,
        token=hub_cloud_dev_token,
    )
    s3_ds.add_creds_key(hub_cloud_dev_managed_creds_key, managed=True)

    assert s3_ds.path.startswith("hub://")
    assert "x" in s3_ds.tensors
    assert s3_ds.x[0].numpy() == 10


def test_connect_dataset_cases(local_ds, memory_ds, hub_cloud_ds):
    # Connecting Local or Memory datasets makes no sense.
    with pytest.raises(InvalidSourcePathError):
        local_ds.connect(creds_key="some_creds", dest_path="hub://someorg/somename")

    with pytest.raises(InvalidSourcePathError):
        memory_ds.connect(creds_key="some_creds", dest_path="hub://someorg/somename")

    # Connecting a cloud dataset is not permitted as its already avaliable via a Deep Lake path
    with pytest.raises(InvalidSourcePathError):
        hub_cloud_ds.connect(creds_key="some_creds", dest_path="hub://someorg/somename")
