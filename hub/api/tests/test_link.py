import hub
import pytest
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
