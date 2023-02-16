from deeplake.util.exceptions import ReadOnlyModeError, EmptyTensorError
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
    runner = CliRunner()
    result = runner.invoke(logout)
    assert result.exit_code == 0

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


def test_view_public(hub_cloud_dev_credentials):
    runner = CliRunner()
    result = runner.invoke(logout)
    assert result.exit_code == 0

    username, password = hub_cloud_dev_credentials

    ds = deeplake.load("hub://activeloop/mnist-train")
    view = ds[100:200]

    with pytest.raises(ReadOnlyModeError):
        view.save_view(id="100to200")

    runner.invoke(login, f"-u {username} -p {password}")

    ds = deeplake.load("hub://activeloop/mnist-train")
    view = ds[100:200]

    with pytest.raises(ReadOnlyModeError):
        view.save_view(id="100to200")

    runner.invoke(logout)


def test_view_with_empty_tensor(local_ds):
    with local_ds as ds:
        ds.create_tensor("images")
        ds.images.extend([1, 2, 3, 4, 5])

        ds.create_tensor("labels")
        ds.labels.extend([None, None, None, None, None])
        ds.commit()

        ds[:3].save_view(id="save1", optimize=True)

    view = ds.load_view("save1")

    assert len(view) == 3

    with pytest.raises(EmptyTensorError):
        view.labels.numpy()

    np.testing.assert_array_equal(
        view.images.numpy(), np.array([1, 2, 3]).reshape(3, 1)
    )
