import pickle
import hub
import pytest
from hub.constants import GCS_OPT, S3_OPT
from hub.core.link_creds import LinkCreds
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
        link_creds.get_storage_provider("abc", "random_provider")
    with pytest.raises(ValueError):
        link_creds.get_storage_provider(None, "random_provider")
    with pytest.raises(ValueError):
        link_creds.get_storage_provider("ghi", "s3")

    if is_opt_true(request, GCS_OPT):
        assert isinstance(link_creds.get_storage_provider("def", "gcs"), GCSProvider)
        assert isinstance(link_creds.get_storage_provider("ENV", "gcs"), GCSProvider)
    if is_opt_true(request, S3_OPT):
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
    assert len(bts) == link_creds.nbytes()
    assert bts == b"abc,def,ghi"

    from_buffer_link_creds = LinkCreds.frombuffer(bts)
    assert len(from_buffer_link_creds) == 3
    assert from_buffer_link_creds.missing_keys == ["abc", "def", "ghi"]


@pytest.mark.parametrize("create_shape_tensor", [True, False])
@pytest.mark.parametrize("verify", [True, False])
def test_basic(local_ds, cat_path, flower_path, create_shape_tensor, verify):
    with local_ds as ds:
        ds.create_tensor(
            "linked_images",
            htype="link[image]",
            create_shape_tensor=create_shape_tensor,
            verify=verify,
            sample_compression="jpeg",
        )
        for i in range(10):
            sample = hub.link(cat_path) if i % 2 == 0 else hub.link(flower_path)
            ds.linked_images.append(sample)
        assert len(ds.linked_images) == 10
        for i in range(10):
            shape_target = (900, 900, 3) if i % 2 == 0 else (513, 464, 4)
            assert ds.linked_images[i].shape == shape_target
            assert ds.linked_images[i].numpy().shape == shape_target


@pytest.mark.parametrize("create_shape_tensor", [True, False])
@pytest.mark.parametrize("verify", [True, False])
def test_basic_sequence(local_ds, cat_path, flower_path, create_shape_tensor, verify):
    pass


def test_basic_add_populate_creds():
    pass
