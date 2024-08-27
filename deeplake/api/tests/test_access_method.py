import deeplake
import pytest
import numpy as np
from deeplake.util.exceptions import DatasetHandlerError
from deeplake.util.access_method import parse_access_method, get_local_storage_path


def test_local_storage_path():
    path = "/tmp/deeplake/"
    dataset_name = "hub://activeloop/mnist-train"
    assert (
        get_local_storage_path(dataset_name, path)
        == "/tmp/deeplake/hub_activeloop_mnist-train"
    )


def test_access_method_parsing():
    assert parse_access_method("download") == ("download", 0, "threaded")
    assert parse_access_method("download:processed") == ("download", 0, "processed")
    assert parse_access_method("download:5") == ("download", 5, "threaded")
    assert parse_access_method("download:processed:5") == ("download", 5, "processed")
    assert parse_access_method("download:5:processed") == ("download", 5, "processed")
    assert parse_access_method("local") == ("local", 0, "threaded")
    assert parse_access_method("local:processed") == ("local", 0, "processed")
    assert parse_access_method("local:5") == ("local", 5, "threaded")
    assert parse_access_method("local:processed:5") == ("local", 5, "processed")
    assert parse_access_method("local:5:processed") == ("local", 5, "processed")
    with pytest.raises(ValueError):
        parse_access_method("download:5:processed:5")
    with pytest.raises(ValueError):
        parse_access_method("download:processed:processed")
    with pytest.raises(ValueError):
        parse_access_method("download:processed:5:processed")
    with pytest.raises(ValueError):
        parse_access_method("local:5:processed:5")
    with pytest.raises(ValueError):
        parse_access_method("local:processed:processed")
    with pytest.raises(ValueError):
        parse_access_method("local:processed:5:processed")


@pytest.mark.slow
def test_access_method(s3_ds_generator):
    with pytest.raises(DatasetHandlerError):
        deeplake.dataset("./some_non_existent_path", access_method="download")

    with pytest.raises(DatasetHandlerError):
        deeplake.dataset("./some_non_existent_path", access_method="local")

    with s3_ds_generator() as ds:
        ds.create_tensor("x")
        ds.x.extend(list(range(10)))

    ds = s3_ds_generator(access_method="local:2")  # downloads automatically
    assert not ds.path.startswith("s3://")
    for i in range(10):
        assert ds.x[i].numpy() == i

    with s3_ds_generator() as ds:
        ds.x.extend(list(range(10, 20)))

    ds = s3_ds_generator(access_method="local")  # load downloaded
    assert not ds.path.startswith("s3://")
    assert len(ds.x) == 10

    ds = s3_ds_generator(access_method="download")  # download again
    assert len(ds.x) == 20
    for i in range(20):
        assert ds.x[i].numpy() == i

    with pytest.raises(ValueError):
        s3_ds_generator(access_method="invalid")

    with pytest.raises(ValueError):
        s3_ds_generator(access_method="download", overwrite=True)

    with pytest.raises(ValueError):
        s3_ds_generator(access_method="local", overwrite=True)

    ds.delete()


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
def test_access_method_with_creds(
    hub_cloud_ds_generator, hub_cloud_dev_managed_creds_key
):
    with hub_cloud_ds_generator() as ds:
        ds.create_tensor("abc")
        ds.create_tensor("images", htype="link[image]", sample_compression="jpg")

        ds.add_creds_key(hub_cloud_dev_managed_creds_key, managed=True)

        ds.abc.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ds.images.extend(
            [
                deeplake.link(
                    "https://picsum.photos/20/30",
                    creds_key=hub_cloud_dev_managed_creds_key,
                )
                for _ in range(10)
            ]
        )

    ds = hub_cloud_ds_generator(access_method="download:2")

    assert ds.images.htype == "link[image]"
    assert ds.images.shape == (10, 30, 20, 3)
    np.testing.assert_array_equal(ds.abc.numpy(), np.arange(1, 11).reshape(-1, 1))

    ds.delete()

    ds = hub_cloud_ds_generator(access_method="download:2", unlink=True)

    assert ds.images.htype == "image"
    assert ds.images.shape == (10, 30, 20, 3)
    np.testing.assert_array_equal(ds.abc.numpy(), np.arange(1, 11).reshape(-1, 1))

    ds.delete()
