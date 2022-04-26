import numpy as np
import os
import sys
import pickle
import hub
import pytest
from hub.constants import GCS_OPT, S3_OPT
from hub.core.link_creds import LinkCreds
from hub.core.meta.encode.creds import CredsEncoder
from hub.core.storage.gcs import GCSProvider
from hub.core.storage.s3 import S3Provider
from hub.tests.common import is_opt_true
from hub.util.exceptions import TensorMetaInvalidHtype

from hub.util.htype import parse_complex_htype  # type: ignore


def test_complex_htype_parsing():
    is_sequence, is_link, htype = parse_complex_htype("link")
    assert not is_sequence
    assert is_link
    assert htype == "generic"

    is_sequence, is_link, htype = parse_complex_htype("sequence")
    assert is_sequence
    assert not is_link
    assert htype == "generic"

    is_sequence, is_link, htype = parse_complex_htype("sequence[link]")
    assert is_sequence
    assert is_link
    assert htype == "generic"

    is_sequence, is_link, htype = parse_complex_htype("link[sequence]")
    assert is_sequence
    assert is_link
    assert htype == "generic"

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
    link_creds.add_creds("abc")
    link_creds.add_creds("def")

    with pytest.raises(ValueError):
        link_creds.add_creds("abc")

    link_creds.populate_creds("abc", {})
    link_creds.populate_creds("def", {})

    with pytest.raises(KeyError):
        link_creds.populate_creds("ghi", {})

    assert link_creds.get_encoding("ENV") == 0
    assert link_creds.get_encoding(None) == 0
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

    link_creds.add_creds("ghi")
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
    assert bts == b"abc,def,ghi"

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
        ds.add_creds("my_s3_key")
        ds.add_creds("my_gcs_key")
        ds.populate_creds("my_s3_key", {})
        ds.populate_creds("my_gcs_key", {})

        assert ds.link_creds.creds_keys == ["my_s3_key", "my_gcs_key"]
        assert ds.link_creds.creds_mapping == {"my_s3_key": 1, "my_gcs_key": 2}
        assert ds.link_creds.creds_dict == {"my_s3_key": {}, "my_gcs_key": {}}

    ds = local_ds_generator()
    assert ds.link_creds.creds_keys == ["my_s3_key", "my_gcs_key"]
    assert ds.link_creds.creds_mapping == {"my_s3_key": 1, "my_gcs_key": 2}
    assert ds.link_creds.creds_dict == {}


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
            sample_compression="jpeg",
        )
        with pytest.raises(TypeError):
            ds.linked_images.append(np.ones((100, 100, 3)))

        for i in range(10):
            sample = hub.link(cat_path) if i % 2 == 0 else hub.link(flower_path)
            ds.linked_images.append(sample)

        # Uncomment after text is fixed
        # for _ in range(10):
        #     sample = hub.link(flower_path)
        #     ds.linked_images.append(sample)

        # for i in range(0, 10, 2):
        #     sample = hub.link(cat_path)
        #     ds.linked_images[i] = sample

        assert len(ds.linked_images) == 10

        ds.create_tensor(
            "linked_images_2",
            htype="link[image]",
            create_shape_tensor=create_shape_tensor,
            verify=verify,
            sample_compression="jpeg",
        )
        ds.linked_images_2.extend(ds.linked_images)
        assert len(ds.linked_images_2) == 10
        for i in range(10):
            shape_target = (900, 900, 3) if i % 2 == 0 else (513, 464, 4)
            assert ds.linked_images[i].shape == shape_target
            assert ds.linked_images[i].numpy().shape == shape_target
            assert ds.linked_images_2[i].shape == shape_target
            assert ds.linked_images_2[i].numpy().shape == shape_target

    # checking persistence
    ds = local_ds_generator()
    for i in range(10):
        shape_target = (900, 900, 3) if i % 2 == 0 else (513, 464, 4)
        assert ds.linked_images[i].shape == shape_target
        assert ds.linked_images[i].numpy().shape == shape_target
        assert ds.linked_images_2[i].shape == shape_target
        assert ds.linked_images_2[i].numpy().shape == shape_target


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
            create_shape_tensor=create_shape_tensor,
            verify=verify,
        )
        for _ in range(3):
            sample = hub.link(
                "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
            )
            ds.linked_videos.append(sample)

        assert len(ds.linked_videos) == 3
        for i in range(3):
            assert ds.linked_videos[i].shape == (361, 720, 1280, 3)
            assert ds.linked_videos[i][:5].numpy().shape == (5, 720, 1280, 3)

        if is_opt_true(request, GCS_OPT):
            sample = hub.link("gcs://gtv-videos-bucket/sample/ForBiggerJoyrides.mp4")
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
            verify=False,
            create_shape_tensor=False,
            create_sample_info_tensor=False,
        )
        ds.create_tensor("xyz")
        ds.add_creds("my_first_key")
        ds.add_creds("my_second_key")

        ds.populate_creds("my_first_key", {})
        ds.populate_creds("my_second_key", {})
        for i in range(10):
            creds_key = "my_first_key" if i % 2 == 0 else "my_second_key"
            sample = hub.link("https://picsum.photos/200/300", creds_key=creds_key)
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
            sample = hub.link("https://picsum.photos/200/300")
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


@hub.compute
def identity(sample_in, samples_out):
    samples_out.linked_images.append(sample_in.linked_images)


def test_transform(local_ds, cat_path, flower_path):
    data_in = hub.dataset("./test/link_transform", overwrite=True)
    with data_in as ds:
        ds.create_tensor(
            "linked_images",
            htype="link[image]",
            create_shape_tensor=True,
            verify=True,
            sample_compression="jpeg",
        )
        for i in range(10):
            sample = hub.link(cat_path) if i % 2 == 0 else hub.link(flower_path)
            ds.linked_images.append(sample)

    data_out = local_ds
    with data_out as ds:
        ds.create_tensor(
            "linked_images",
            htype="link[image]",
            create_shape_tensor=True,
            verify=True,
            sample_compression="jpeg",
        )

    identity().eval(data_in, data_out, num_workers=2)
    assert len(data_out.linked_images) == 10
    for i in range(10):
        shape_target = (900, 900, 3) if i % 2 == 0 else (513, 464, 4)
        assert ds.linked_images[i].shape == shape_target
        assert ds.linked_images[i].numpy().shape == shape_target

    data_in.delete()


@pytest.mark.parametrize("create_shape_tensor", [True, False])
@pytest.mark.parametrize("verify", [True, False])
def test_basic_sequence(local_ds, cat_path, flower_path, create_shape_tensor, verify):
    pass
