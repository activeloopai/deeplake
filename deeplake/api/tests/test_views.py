from deeplake.util.exceptions import DatasetViewSavingError
from deeplake.client.utils import get_user_name
from deeplake.cli.auth import logout, login
from click.testing import CliRunner

import numpy as np

import posixpath
import deeplake
import pytest


def populate(ds):
    ds.create_tensor("images", htype="image", sample_compression="jpg")
    ds.create_tensor("labels", htype="class_label")

    ds.extend(
        {
            "images": np.random.randint(0, 256, (100, 20, 20, 3), dtype=np.uint8),
            "labels": np.random.randint(0, 3, (100,)),
        }
    )
    ds.commit()


def test_view_token_only(
    hub_cloud_path, hub_cloud_dev_token, hub_cloud_dev_credentials
):
    username = get_user_name()
    if username != "public":
        state = "logged in"
    else:
        state = "logged out"
    runner = CliRunner()
    result = runner.invoke(logout)
    assert result.exit_code == 0

    username, password = hub_cloud_dev_credentials

    ds = deeplake.empty(hub_cloud_path, token=hub_cloud_dev_token)
    with ds:
        populate(ds)

    ds = deeplake.load(hub_cloud_path, token=hub_cloud_dev_token)
    view = ds[50:100]
    view.save_view(id="50to100")

    ds = deeplake.load(hub_cloud_path, read_only=True, token=hub_cloud_dev_token)
    view = ds[25:100]
    view.save_view(id="25to100")

    ds = deeplake.load(hub_cloud_path, read_only=True, token=hub_cloud_dev_token)

    loaded = ds.load_view("50to100")
    np.testing.assert_array_equal(loaded.images.numpy(), ds[50:100].images.numpy())
    np.testing.assert_array_equal(loaded.labels.numpy(), ds[50:100].labels.numpy())
    assert loaded._vds.path == posixpath.join(hub_cloud_path, ".queries/50to100")

    loaded = ds.load_view("25to100")
    np.testing.assert_array_equal(loaded.images.numpy(), ds[25:100].images.numpy())
    np.testing.assert_array_equal(loaded.labels.numpy(), ds[25:100].labels.numpy())
    assert loaded._vds.path == posixpath.join(hub_cloud_path, ".queries/25to100")

    ds.delete_view("25to100")
    deeplake.delete(hub_cloud_path, token=hub_cloud_dev_token)

    if state == "logged in":
        runner.invoke(login, f"-u {username} -p {password}")


def test_view_public(hub_cloud_dev_credentials):
    username = get_user_name()
    if username != "public":
        state = "logged in"
    else:
        state = "logged out"
    runner = CliRunner()
    result = runner.invoke(logout)
    assert result.exit_code == 0

    username, password = hub_cloud_dev_credentials

    ds = deeplake.load("hub://activeloop/mnist-train")
    view = ds[100:200]

    # not logged in
    with pytest.raises(DatasetViewSavingError):
        view.save_view(id="100to200")

    runner.invoke(login, f"-u {username} -p {password}")

    ds = deeplake.load("hub://activeloop/mnist-train")
    view = ds[100:200]
    view.save_view(id="100to200")

    runner.invoke(logout)
    ds = deeplake.load("hub://activeloop/mnist-train")
    with pytest.raises(KeyError):
        ds.load_view("100to200")

    runner.invoke(login, f"-u {username} -p {password}")

    ds = deeplake.load("hub://activeloop/mnist-train")
    loaded = ds.load_view("100to200")
    np.testing.assert_array_equal(loaded.images.numpy(), ds[100:200].images.numpy())
    np.testing.assert_array_equal(loaded.labels.numpy(), ds[100:200].labels.numpy())
    assert (
        loaded._vds.path
        == f"hub://{username}/queries/.queries/[activeloop][mnist-train]100to200"
    )

    ds.delete_view("100to200")

    if state == "logged out":
        runner.invoke(logout)
