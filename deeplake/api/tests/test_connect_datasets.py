import os

import pytest

import deeplake
from deeplake.util.exceptions import InvalidSourcePathError, TokenPermissionError


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_connect_dataset_cases(local_ds, memory_ds, hub_cloud_ds):
    # Connecting Local or Memory datasets makes no sense.
    with pytest.raises(InvalidSourcePathError):
        local_ds.connect(creds_key="some_creds", dest_path="hub://someorg/somename")

    with pytest.raises(InvalidSourcePathError):
        memory_ds.connect(creds_key="some_creds", dest_path="hub://someorg/somename")

    # Connecting a cloud dataset is not permitted as its already avaliable via a Deep Lake path
    with pytest.raises(InvalidSourcePathError):
        hub_cloud_ds.connect(creds_key="some_creds", dest_path="hub://someorg/somename")


@pytest.mark.slow
def test_connect_user_not_in_org(s3_ds_generator, hub_cloud_dev_token):
    with s3_ds_generator() as ds:
        ds.create_tensor("x")
        ds.x.append(10)

    with pytest.raises(TokenPermissionError) as e:
        ds.connect(
            creds_key="some_creds",
            dest_path="hub://bad-org/some-name",
            token=hub_cloud_dev_token,
        )
        assert "dataset path" in str(e)

    with pytest.raises(TokenPermissionError) as e:
        ds.connect(
            creds_key="some_creds",
            org_id="bad-org",
            ds_name="some-name",
            token=hub_cloud_dev_token,
        )
        assert "organization id" in str(e)


# @pytest.mark.slow
def test_connect_from_managed_credentials(hub_cloud_path: str, hub_cloud_dev_token):
    old_environ = dict(os.environ)
    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    os.environ.pop("AWS_SESSION_TOKEN", None)

    try:
        dir_name = hub_cloud_path.rsplit("/", 1)[1]
        ds = deeplake.empty(
            f"s3://deeplake-tests/{dir_name}",
            creds={"creds_key": "aws_creds"},
            org_id="testingacc2",
            token=hub_cloud_dev_token,
        )
        ds.create_tensor("id", htype="text")

        ds.connect()
        assert ds.path == f"hub://testingacc2/{dir_name}"

    finally:
        os.environ.clear()
        os.environ.update(old_environ)
