import numpy as np
import os
import sys
import pickle
import deeplake
import pytest
from deeplake.client.client import DeepLakeBackendClient
from deeplake.constants import GCS_OPT, S3_OPT
from deeplake.core.link_creds import LinkCreds
from deeplake.core.meta.encode.creds import CredsEncoder
from deeplake.core.storage.gcs import GCSProvider
from deeplake.core.storage.s3 import S3Provider
from deeplake.tests.common import is_opt_true
from deeplake.util.exceptions import (
    ManagedCredentialsNotFoundError,
    TensorMetaInvalidHtype,
    UnableToReadFromUrlError,
)

from deeplake.util.htype import parse_complex_htype  # type: ignore


def test_complex_htype_parsing():
    with pytest.raises(ValueError):
        is_sequence, is_link, htype = parse_complex_htype("link")

    is_sequence, is_link, htype = parse_complex_htype("sequence")
    assert is_sequence
    assert not is_link
    assert htype == "generic"

    with pytest.raises(ValueError):
        is_sequence, is_link, htype = parse_complex_htype("sequence[link]")

    with pytest.raises(ValueError):
        is_sequence, is_link, htype = parse_complex_htype("link[sequence]")

    is_sequence, is_link, htype = parse_complex_htype("sequence[image]")
    assert is_sequence
    assert not is_link
    assert htype == "image"

    is_sequence, is_link, htype = parse_complex_htype("link[image]")
    assert not is_sequence
    assert is_link
    assert htype == "image"

    is_sequence, is_link, htype = parse_complex_htype("link[sequence[image]]")
    assert is_sequence
    assert is_link
    assert htype == "image"

    is_sequence, is_link, htype = parse_complex_htype("sequence[link[video]]")
    assert is_sequence
    assert is_link
    assert htype == "video"

    bad_inputs = [
        "random[image]",
        "sequence[random[image]]",
        "link[random[image]]",
        "link(image)",
        "sequence(image)",
        "link[sequence(image)]",
        "sequence[link(image)]",
        "link[sequence[image[uint8]]]",
        "sequence[link[image[uint8]]]",
    ]
    for bad_input in bad_inputs:
        with pytest.raises(TensorMetaInvalidHtype):
            parse_complex_htype(bad_input)


def test_link_creds(request):
    link_creds = LinkCreds()
    link_creds.add_creds_key("abc")
    link_creds.add_creds_key("def")

    with pytest.raises(ValueError):
        link_creds.add_creds_key("abc")

    link_creds.populate_creds("abc", {})
    link_creds.populate_creds("def", {})

    with pytest.raises(KeyError):
        link_creds.populate_creds("ghi", {})

    assert link_creds.get_encoding("ENV") == 0
    assert link_creds.get_encoding(None) == 0
    with pytest.raises(ValueError):
        link_creds.get_encoding(None, "s3://my_bucket/my_key")
    assert link_creds.get_encoding("abc") == 1
    assert link_creds.get_encoding("def") == 2
    with pytest.raises(ValueError):
        link_creds.get_encoding("ghi")

    assert link_creds.get_creds_key(0) is None
    assert link_creds.get_creds_key(1) == "abc"
    assert link_creds.get_creds_key(2) == "def"
    with pytest.raises(KeyError):
        link_creds.get_creds_key(3)

    assert len(link_creds) == 2
    assert link_creds.missing_keys == []

    link_creds.add_creds_key("ghi")
    assert link_creds.missing_keys == ["ghi"]

    with pytest.raises(KeyError):
        link_creds.get_storage_provider("xyz", "s3")

    with pytest.raises(ValueError):
        link_creds.get_storage_provider("ghi", "s3")

    if is_opt_true(request, GCS_OPT):
        assert isinstance(link_creds.get_storage_provider("def", "gcs"), GCSProvider)
        assert isinstance(link_creds.get_storage_provider("def", "gcs"), GCSProvider)
        assert isinstance(link_creds.get_storage_provider("ENV", "gcs"), GCSProvider)
    if is_opt_true(request, S3_OPT):
        assert isinstance(link_creds.get_storage_provider("abc", "s3"), S3Provider)
        assert isinstance(link_creds.get_storage_provider("abc", "s3"), S3Provider)
        assert isinstance(link_creds.get_storage_provider(None, "s3"), S3Provider)

    pickled = pickle.dumps(link_creds)
    unpickled_link_creds = pickle.loads(pickled)

    assert len(unpickled_link_creds) == 3
    assert unpickled_link_creds.get_creds_key(0) is None
    assert unpickled_link_creds.get_creds_key(1) == "abc"
    assert unpickled_link_creds.get_creds_key(2) == "def"
    assert unpickled_link_creds.missing_keys == ["ghi"]

    bts = link_creds.tobytes()
    assert len(bts) == link_creds.nbytes

    from_buffer_link_creds = LinkCreds.frombuffer(bts)
    assert len(from_buffer_link_creds) == 3
    assert from_buffer_link_creds.missing_keys == ["abc", "def", "ghi"]


def test_creds_encoder():
    enc = CredsEncoder()

    enc.register_samples((3,), 5)
    enc.register_samples((0,), 2)
    enc.register_samples((1,), 3)

    for i in range(5):
        assert enc[i] == (3,)
    for i in range(5, 7):
        assert enc[i] == (0,)
    for i in range(7, 10):
        assert enc[i] == (1,)

    data = enc.tobytes()
    assert len(data) == enc.nbytes

    dec = CredsEncoder.frombuffer(data)
    for i in range(5):
        assert dec[i] == (3,)
    for i in range(5, 7):
        assert dec[i] == (0,)
    for i in range(7, 10):
        assert dec[i] == (1,)


def test_add_populate_creds(local_ds_generator):
    local_ds = local_ds_generator()
    with local_ds as ds:
        ds.add_creds_key("my_s3_key")
        ds.add_creds_key("my_gcs_key")
        ds.populate_creds("my_s3_key", {})
        ds.populate_creds("my_gcs_key", {})

        assert ds.link_creds.creds_keys == ["my_s3_key", "my_gcs_key"]
        assert ds.link_creds.creds_mapping == {"my_s3_key": 1, "my_gcs_key": 2}
        assert ds.link_creds.creds_dict == {"my_s3_key": {}, "my_gcs_key": {}}

    ds = local_ds_generator()
    assert ds.link_creds.creds_keys == ["my_s3_key", "my_gcs_key"]
    assert ds.link_creds.creds_mapping == {"my_s3_key": 1, "my_gcs_key": 2}
    assert ds.link_creds.creds_dict == {}


def test_none_used_key(local_ds_generator, cat_path):
    local_ds = local_ds_generator()
    with local_ds as ds:
        ds.create_tensor("xyz", htype="link[image]", sample_compression="jpg")
        ds.add_creds_key("my_s3_key")
        ds.populate_creds("my_s3_key", {})
        ds.xyz.append(deeplake.link(cat_path))
        assert ds.link_creds.used_creds_keys == set()
        ds.xyz.append(deeplake.link(cat_path, "ENV"))
        assert ds.link_creds.used_creds_keys == set()
        ds.xyz.append(deeplake.link(cat_path, "my_s3_key"))
        assert ds.link_creds.used_creds_keys == {"my_s3_key"}

    ds = local_ds_generator()
    assert ds.link_creds.used_creds_keys == {"my_s3_key"}


@pytest.mark.parametrize("create_shape_tensor", [True, False])
@pytest.mark.parametrize("verify", [True, False])
def test_basic(local_ds_generator, cat_path, flower_path, create_shape_tensor, verify):
    local_ds = local_ds_generator()
    with local_ds as ds:
        ds.create_tensor(
            "linked_images",
            htype="link[image]",
            create_shape_tensor=create_shape_tensor,
            verify=verify,
            sample_compression="png",
        )
        with pytest.raises(TypeError):
            ds.linked_images.append(np.ones((100, 100, 3)))

        for _ in range(10):
            sample = deeplake.link(flower_path)
            ds.linked_images.append(sample)
        assert ds.linked_images.meta.sample_compression == "png"

        ds.linked_images.append(None)

        for i in range(0, 10, 2):
            sample = deeplake.link(cat_path)
            ds.linked_images[i] = sample

        assert len(ds.linked_images) == 11

        ds.create_tensor(
            "linked_images_2",
            htype="link[image]",
            create_shape_tensor=create_shape_tensor,
            verify=verify,
            sample_compression="png",
        )
        ds.linked_images_2.extend(ds.linked_images)
        assert len(ds.linked_images_2) == 11
        for i in range(10):
            shape_target = (900, 900, 3) if i % 2 == 0 else (513, 464, 4)
            assert ds.linked_images[i].shape == shape_target
            assert ds.linked_images[i].numpy().shape == shape_target
            assert ds.linked_images_2[i].shape == shape_target
            assert ds.linked_images_2[i].numpy().shape == shape_target

        assert ds.linked_images_2.meta.sample_compression == "png"
        assert ds.linked_images[10].size == 0
        np.testing.assert_array_equal(ds.linked_images[10].numpy(), np.ones((0,)))
        assert ds.linked_images_2[10].size == 0
        np.testing.assert_array_equal(ds.linked_images_2[10].numpy(), np.ones((0,)))

    ds.commit()
    view = ds[:5]
    view.save_view(optimize=True)
    view2 = ds.get_views()[0].load()
    view1_np = view.linked_images.numpy(aslist=True)
    view2_np = view2.linked_images.numpy(aslist=True)

    assert len(view1_np) == len(view2_np)
    for v1, v2 in zip(view1_np, view2_np):
        np.testing.assert_array_equal(v1, v2)

    view_id = ds[:10].save_view().split("queries/")[1]
    view3 = ds.load_view(view_id, optimize=True)
    assert view3.linked_images.meta.sample_compression == "png"
    assert view3.linked_images_2.meta.sample_compression == "png"

    # checking persistence
    ds = local_ds_generator()
    for i in range(10):
        shape_target = (900, 900, 3) if i % 2 == 0 else (513, 464, 4)
        assert ds.linked_images[i].shape == shape_target
        assert ds.linked_images[i].numpy().shape == shape_target
        assert ds.linked_images_2[i].shape == shape_target
        assert ds.linked_images_2[i].numpy().shape == shape_target


@pytest.mark.xfail(reason="broken link")
def test_jwt_link(local_ds):
    with local_ds as ds:
        ds.create_tensor(
            "img",
            htype="link[image]",
            sample_compression="jpg",
            create_shape_tensor=False,
        )
        auth = DeepLakeBackendClient().auth_header
        my_jwt = {"Authorization": auth}
        ds.add_creds_key("my_jwt_key")
        ds.populate_creds("my_jwt_key", my_jwt)
        img_url = "https://app-dev.activeloop.dev/api/org/tim4/storage/image"
        for _ in range(3):
            ds.img.append(deeplake.link(img_url, creds_key="my_jwt_key"))

        for i in range(3):
            assert ds.img[i].shape == (50, 50, 4)
            assert ds.img[i].numpy().shape == (50, 50, 4)

        my_incorrect_jwt = {"Authorization": "12345"}
        ds.populate_creds("my_jwt_key", my_incorrect_jwt)
        with pytest.raises(UnableToReadFromUrlError):
            ds.img[0].numpy()

        with pytest.raises(UnableToReadFromUrlError):
            ds.img[0].shape


@pytest.mark.parametrize("create_shape_tensor", [True, False])
@pytest.mark.parametrize("verify", [True, False])
@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
def test_video(request, local_ds_generator, create_shape_tensor, verify):
    local_ds = local_ds_generator()
    with local_ds as ds:
        ds.create_tensor(
            "linked_videos",
            htype="link[video]",
            sample_compression="mp4",
            create_shape_tensor=create_shape_tensor,
            verify=verify,
        )
        for _ in range(3):
            sample = deeplake.link(
                "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
            )
            ds.linked_videos.append(sample)

        assert len(ds.linked_videos) == 3
        for i in range(3):
            assert ds.linked_videos[i].shape == (361, 720, 1280, 3)
            assert ds.linked_videos[i][:5].numpy().shape == (5, 720, 1280, 3)

        if is_opt_true(request, GCS_OPT):
            sample = deeplake.link(
                "gcs://gtv-videos-bucket/sample/ForBiggerJoyrides.mp4", creds_key="ENV"
            )
            ds.linked_videos.append(sample)
            assert len(ds.linked_videos) == 4
            assert ds.linked_videos[3].shape == (361, 720, 1280, 3)
    # checking persistence
    ds = local_ds_generator()
    for i in range(3):
        assert ds.linked_videos[i].shape == (361, 720, 1280, 3)

    if is_opt_true(request, GCS_OPT):
        assert len(ds.linked_videos) == 4
        assert ds.linked_videos[3].shape == (361, 720, 1280, 3)


def test_complex_creds(local_ds_generator):
    local_ds = local_ds_generator()
    with local_ds as ds:
        ds.create_tensor(
            "link",
            htype="link[image]",
            sample_compression="jpg",
            verify=False,
            create_shape_tensor=False,
            create_sample_info_tensor=False,
        )
        ds.create_tensor("xyz")
        ds.add_creds_key("my_first_key")
        ds.add_creds_key("my_second_key")

        assert ds.get_creds_keys() == ["my_first_key", "my_second_key"]

        ds.populate_creds("my_first_key", {})
        ds.populate_creds("my_second_key", {})
        for i in range(10):
            creds_key = "my_first_key" if i % 2 == 0 else "my_second_key"
            sample = deeplake.link("https://picsum.photos/200/300", creds_key=creds_key)
            ds.link.append(sample)
            ds.xyz.append(i)

        with pytest.raises(ValueError):
            ds.link._linked_sample()

        with pytest.raises(ValueError):
            ds.xyz[0]._linked_sample()

        linked_sample = ds.link[0]._linked_sample()
        assert linked_sample.path == "https://picsum.photos/200/300"
        assert linked_sample.creds_key == "my_first_key"

        for i in range(10, 15):
            sample = deeplake.link("https://picsum.photos/200/300")
            ds.link.append(sample)
            ds.xyz.append(i)

        for i in range(10):
            enc_creds = 1 if i % 2 == 0 else 2
            assert ds.link.chunk_engine.creds_encoder[i][0] == enc_creds

        for i in range(10, 15):
            assert ds.link.chunk_engine.creds_encoder[i][0] == 0

        for i in range(15):
            assert ds.xyz[i].numpy() == i
            assert ds.link[i].numpy().shape == (300, 200, 3)

    ds = local_ds_generator()
    for i in range(10):
        enc_creds = 1 if i % 2 == 0 else 2
        assert ds.link.chunk_engine.creds_encoder[i][0] == enc_creds

    for i in range(10, 15):
        assert ds.link.chunk_engine.creds_encoder[i][0] == 0

    for i in range(15):
        assert ds.xyz[i].numpy() == i

    with pytest.raises(ValueError):
        ds.link[0].numpy().shape


@deeplake.compute
def identity(sample_in, samples_out):
    samples_out.linked_images.append(sample_in.linked_images)


def test_transform(local_ds, cat_path, flower_path):
    data_in = deeplake.dataset("./test/link_transform", overwrite=True)
    with data_in as ds:
        ds.create_tensor(
            "linked_images",
            htype="link[image]",
            create_shape_tensor=True,
            verify=True,
            sample_compression="jpeg",
        )
        for i in range(10):
            sample = (
                deeplake.link(cat_path) if i % 2 == 0 else deeplake.link(flower_path)
            )
            ds.linked_images.append(sample)
        assert ds.linked_images.meta.sample_compression == "jpeg"

    data_out = local_ds
    with data_out as ds:
        ds.create_tensor(
            "linked_images",
            htype="link[image]",
            create_shape_tensor=True,
            verify=True,
            sample_compression="jpeg",
        )
        assert ds.linked_images.meta.sample_compression == "jpeg"

    identity().eval(data_in, data_out, num_workers=2)
    assert len(data_out.linked_images) == 10
    for i in range(10):
        shape_target = (900, 900, 3) if i % 2 == 0 else (513, 464, 4)
        assert ds.linked_images[i].shape == shape_target
        assert ds.linked_images[i].numpy().shape == shape_target

    data_in.delete()


@deeplake.compute
def transform_path_link(sample_in, samples_out):
    samples_out.images.append(deeplake.link(sample_in))


def check_transformed_ds(ds):
    assert ds.images[0].numpy().shape == ds.images[0].shape == (900, 900, 3)
    assert ds.images[1].numpy().shape == ds.images[1].shape == (513, 464, 4)


def test_transform_2(local_ds_generator, cat_path, flower_path):
    ds = local_ds_generator()
    with ds:
        ds.create_tensor("images", htype="link[image]", sample_compression="jpg")

    transform_path_link().eval([cat_path, flower_path], ds)

    check_transformed_ds(ds)
    ds = local_ds_generator()
    check_transformed_ds(ds)


def test_link_managed(hub_cloud_ds_generator, cat_path):
    key_name = "CREDS_MANAGEMENT_TEST"
    with hub_cloud_ds_generator() as ds:
        ds.create_tensor(
            "img",
            htype="link[image]",
            sample_compression="jpg",
            verify=False,
            create_shape_tensor=False,
            create_sample_info_tensor=False,
        )
        ds.add_creds_key(key_name, managed=True)
        assert key_name in ds.link_creds.creds_dict
        assert key_name in ds.link_creds.managed_creds_keys
        assert key_name not in ds.link_creds.used_creds_keys

        ds.img.append(deeplake.link(cat_path, creds_key=key_name))
        assert key_name in ds.link_creds.used_creds_keys

    ds = hub_cloud_ds_generator()
    assert key_name in ds.link_creds.creds_dict
    assert key_name in ds.link_creds.managed_creds_keys
    assert key_name in ds.link_creds.used_creds_keys

    shape_target = (900, 900, 3)
    assert ds.img[0].shape == shape_target
    assert ds.img[0].numpy().shape == shape_target

    with pytest.raises(ValueError):
        # managed creds_key can't be updated
        ds.update_creds_key(key_name, "something_else")

    with pytest.raises(KeyError):
        ds.change_creds_management("random_key", False)

    # this is a no-op
    ds.change_creds_management(key_name, True)

    # no longer managed
    ds.change_creds_management(key_name, False)

    ds = hub_cloud_ds_generator()
    with pytest.raises(ValueError):
        ds.img[0].numpy()

    ds.populate_creds(key_name, {})
    assert ds.img[0].shape == shape_target
    assert ds.img[0].numpy().shape == shape_target

    ds = hub_cloud_ds_generator()
    ds.change_creds_management(key_name, True)
    assert ds.img[0].shape == shape_target
    assert ds.img[0].numpy().shape == shape_target

    new_key = "some_random_key"
    with pytest.raises(ManagedCredentialsNotFoundError):
        ds.add_creds_key(new_key, managed=True)

    # even after failure one can simply add a new key, setting managed to False
    ds.add_creds_key(new_key)


def test_link_ready(local_ds_generator, cat_path):
    with local_ds_generator() as ds:
        ds.create_tensor(
            "img",
            htype="link[image]",
            sample_compression="jpg",
            verify=False,
            create_shape_tensor=False,
            create_sample_info_tensor=False,
        )
        ds.add_creds_key("def")
        ds.add_creds_key("abc")
        ds.populate_creds("abc", {})
        ds.img.append(deeplake.link(cat_path, creds_key="abc"))

    ds = local_ds_generator()
    with pytest.raises(ValueError):
        ds.img[0].numpy()
    ds.populate_creds("abc", {})
    assert ds.img[0].numpy().shape == (900, 900, 3)
    with pytest.raises(KeyError):
        ds.update_creds_key("xyz", "ghi")
    with pytest.raises(ValueError):
        ds.update_creds_key("abc", "def")
    ds.update_creds_key("abc", "new")
    assert ds.img[0].numpy().shape == (900, 900, 3)
    ds = local_ds_generator()
    with pytest.raises(ValueError):
        ds.img[0].numpy()
    ds.populate_creds("new", {})
    assert ds.img[0].numpy().shape == (900, 900, 3)


def test_link_path(local_ds):
    with local_ds as ds:
        ds.create_tensor(
            "a",
            htype="link[image]",
            verify=False,
            create_shape_tensor=False,
            create_sample_info_tensor=False,
            sample_compression="jpeg",
        )
        ds.create_tensor("b", htype="text")
        ds.a.append(deeplake.link("hello!!!!"))
        ds.b.append("hello!!!!")
        ds.a.append(deeplake.link("world"))
        ds.b.append("world")
        ds.a.append(deeplake.link("foo"))
        ds.b.append("foo")

        np.testing.assert_array_equal(ds.a[0].path(), ds.b[0].numpy())
        np.testing.assert_array_equal(ds.a[1].path(), ds.b[1].numpy())
        np.testing.assert_array_equal(ds.a[2].path(), ds.b[2].numpy())
        np.testing.assert_array_equal(ds.a.path(), ds.b.numpy())


@pytest.mark.parametrize("create_shape_tensor", [True, False])
@pytest.mark.parametrize("verify", [True, False])
def test_basic_sequence(local_ds, cat_path, flower_path, create_shape_tensor, verify):
    pass
