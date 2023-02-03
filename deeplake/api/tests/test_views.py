from deeplake.client.utils import get_user_name
from click.testing import CliRunner
from deeplake.cli.auth import logout

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
            "label": np.random.randint(0, 3, (100,)),
        }
    )


def test_view_token_only(
    hub_cloud_path, hub_cloud_dev_token, hub_cloud_dev_credentials
):
    username = get_user_name()
    if username != "public":
        login = True
    else:
        login = False
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
    assert loaded.path == posixpath.join(hub_cloud_path, ".queries/50to1000")

    ds = ds.load_view("25to100")
    np.testing.assert_array_equal(loaded.images.numpy(), ds[25:100].images.numpy())
    np.testing.assert_array_equal(loaded.labels.numpy(), ds[25:100].labels.numpy())
    assert loaded.path == f"hub://{username}/queries/.queries/25to100"
