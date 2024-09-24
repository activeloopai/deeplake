from deeplake.util.exceptions import ReadOnlyModeError, EmptyTensorError, TransformError

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


@pytest.mark.slow
def test_view_token_only(
    hub_cloud_path, hub_cloud_dev_token, hub_cloud_dev_credentials
):
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


@pytest.mark.slow
def test_view_public(hub_cloud_dev_token):
    ds = deeplake.load("hub://activeloop/mnist-train")
    view = ds[100:200]

    with pytest.raises(ReadOnlyModeError):
        view.save_view(id="100to200")

    ds = deeplake.load("hub://activeloop/mnist-train", token=hub_cloud_dev_token)
    view = ds[100:200]

    with pytest.raises(ReadOnlyModeError):
        view.save_view(id="100to200")


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


@pytest.mark.slow
def test_vds_read_only(hub_cloud_path, hub_cloud_dev_token):
    ds = deeplake.empty(hub_cloud_path, token=hub_cloud_dev_token)
    with ds:
        ds.create_tensor("abc")
        ds.abc.extend([1, 2, 3, 4, 5])
        ds.commit()

    ds[:3].save_view(id="first_3")

    ds = deeplake.load(hub_cloud_path, read_only=True, token=hub_cloud_dev_token)

    view = ds.load_view("first_3")

    assert view.base_storage.read_only == True
    assert view._vds.base_storage.read_only == True


def test_view_from_different_commit(local_ds):
    with local_ds as ds:
        ds.create_tensor("x")
        ds.x.extend(list(range(10)))
        cid = ds.commit()
        view = ds[4:9]
        view.save_view(id="abcd")
        ds.x.extend(list(range(10, 20)))
        cid2 = ds.commit()
        view2 = ds.load_view("abcd")
        assert view2.commit_id == cid
        assert ds.commit_id == cid2
        assert not view2.is_optimized
        view2.save_view(id="efg", optimize=True)
        view3 = ds.load_view("efg")
        assert ds.commit_id == cid2
        assert view3.is_optimized


@pytest.mark.slow
@pytest.mark.flaky(retry_count=3)
def test_save_view_ignore_errors(local_ds):
    with local_ds as ds:
        ds.create_tensor(
            "images", htype="link[image]", sample_compression="jpg", verify=False
        )
        ds.create_tensor("labels", htype="class_label")

        ds.images.extend(
            [deeplake.link("https://picsum.photos/20/30") for _ in range(8)]
        )
        ds.images.extend([deeplake.link("https://abcd/20") for _ in range(2)])
        ds.images.extend(
            [deeplake.link("https://picsum.photos/20/30") for _ in range(10)]
        )

        ds.labels.extend([0 for _ in range(20)])

        ds.commit()

    with pytest.raises(TransformError):
        ds[:10].save_view(id="one", optimize=True, num_workers=2)

    ds[:10].save_view(id="two", optimize=True, ignore_errors=True, num_workers=2)
    view = ds.load_view("two")

    assert len(view) == 8

    assert view.images.htype == "image"
    assert view.images.shape == (8, 30, 20, 3)

    np.testing.assert_array_equal(view.labels.numpy(), np.array([[0]] * 8))


@pytest.mark.parametrize("optimize_first_view", [True, False])
@pytest.mark.parametrize("optimize_second_view", [True, False])
def test_save_view_of_view(
    local_ds_generator, optimize_first_view, optimize_second_view
):
    with local_ds_generator() as ds:
        ds.create_tensor("abc")
        ds.abc.extend(list(range(100)))

        ds.commit()

        ds[:20].save_view(id="first_20", optimize=optimize_first_view)

        view = ds.load_view("first_20")

        view[:10].save_view(id="first_10", optimize=optimize_second_view)
        view[10:].save_view(id="second_10", optimize=optimize_second_view)

        first_10, second_10 = ds.load_view("first_10"), ds.load_view("second_10")
        np.testing.assert_array_equal(first_10.abc.numpy(), ds[:10].abc.numpy())
        np.testing.assert_array_equal(second_10.abc.numpy(), ds[10:20].abc.numpy())

    with local_ds_generator() as ds:
        first_10, second_10 = ds.load_view("first_10"), ds.load_view("second_10")
        np.testing.assert_array_equal(first_10.abc.numpy(), ds[:10].abc.numpy())
        np.testing.assert_array_equal(second_10.abc.numpy(), ds[10:20].abc.numpy())


@pytest.mark.parametrize("optimize_first_view", [True, False])
@pytest.mark.parametrize("optimize_second_view", [True, False])
@pytest.mark.parametrize("optimize_third_view", [True, False])
def test_save_view_of_view_of_view(
    local_ds_generator, optimize_first_view, optimize_second_view, optimize_third_view
):
    with local_ds_generator() as ds:
        ds.create_tensor("abc")
        ds.abc.extend(list(range(100)))

        ds.commit()

        ds[:20].save_view(id="first_20", optimize=optimize_first_view)

        view = ds.load_view("first_20")

        view[:10].save_view(id="first_10", optimize=optimize_second_view)

        first_10 = ds.load_view("first_10")

        first_10[:5].save_view(id="first_5", optimize=optimize_third_view)

        first_5 = ds.load_view("first_5")
        np.testing.assert_array_equal(first_5.abc.numpy(), ds[:5].abc.numpy())

    with local_ds_generator() as ds:
        first_5 = ds.load_view("first_5")
        np.testing.assert_array_equal(first_5.abc.numpy(), ds[:5].abc.numpy())
