import pytest
import deeplake
from deeplake.util.exceptions import SampleAppendError

WARNING_STR = "Grayscale images will be reshaped"


@pytest.fixture(params=["jpeg"])
def deeplake_read_images(request, grayscale_image_paths, color_image_paths):
    gray_path = grayscale_image_paths[request.param]
    color_path = color_image_paths[request.param]
    yield request.param, deeplake.read(gray_path), deeplake.read(color_path)


def make_tensor_and_append(ds, htype, sample_compression, images):
    ds.create_tensor("images", htype=htype, sample_compression=sample_compression)
    for img in images:
        ds.images.append(img)


def make_tensor_and_extend(ds, htype, sample_compression, images):
    ds.create_tensor("images", htype=htype, sample_compression=sample_compression)
    ds.images.extend(images)


def test_append_grayscale_second(local_ds_generator, deeplake_read_images):
    "Append a deeplake.read color image first, then a grayscale"
    ds = local_ds_generator()
    imgtype, gray, color = deeplake_read_images
    with pytest.warns(UserWarning, match=WARNING_STR):
        make_tensor_and_append(ds, "image", imgtype, [color, gray])
    assert len(ds.images) == 2
    assert ds.images._sample_shape_tensor.shape == (2, 3)
    for i in range(2):
        assert ds.images[i].numpy().shape == ds.images[i].shape
    assert ds.images.meta.min_shape[-1] == 1
    assert ds.images.meta.max_shape[-1] == 3


def test_append_grayscale_second_many(local_ds_generator, deeplake_read_images):
    "Append a deeplake.read color image first, then a mix of color and gray."
    ds = local_ds_generator()
    imgtype, gray, color = deeplake_read_images
    with pytest.warns(UserWarning, match=WARNING_STR):
        make_tensor_and_append(
            ds, "image", imgtype, [color, color, gray, color, gray, color]
        )
    assert len(ds.images) == 6
    assert ds.images._sample_shape_tensor.shape == (6, 3)
    for i in range(6):
        assert ds.images[i].numpy().shape == ds.images[i].shape
    assert ds.images.meta.min_shape[-1] == 1
    assert ds.images.meta.max_shape[-1] == 3


def test_extend_grayscale_second(local_ds_generator, deeplake_read_images):
    "Extend a dataset with a list of color first, gray second."
    ds = local_ds_generator()
    imgtype, gray, color = deeplake_read_images
    with pytest.warns(UserWarning, match=WARNING_STR):
        make_tensor_and_extend(ds, "image", imgtype, [color, gray])
    assert len(ds.images) == 2
    assert ds.images._sample_shape_tensor.shape == (2, 3)
    for i in range(2):
        assert ds.images[i].numpy().shape == ds.images[i].shape
    assert ds.images.meta.min_shape[-1] == 1
    assert ds.images.meta.max_shape[-1] == 3


def test_append_grayscale_first(local_ds_generator, deeplake_read_images):
    "Append a gray first, color second."
    ds = local_ds_generator()
    imgtype, gray, color = deeplake_read_images
    make_tensor_and_append(ds, "image", imgtype, [gray, color])


def test_append_grayscale_second_generic_ds(local_ds_generator, deeplake_read_images):
    "Append with htype=generic, sample_compression=<valid image compression>."
    ds = local_ds_generator()
    imgtype, gray, color = deeplake_read_images
    with pytest.warns(UserWarning, match=WARNING_STR):
        make_tensor_and_append(ds, "generic", imgtype, [color, gray])
    assert ds.images._sample_shape_tensor.shape == (2, 3)
    for i in range(2):
        assert ds.images[i].numpy().shape == ds.images[i].shape


@pytest.mark.xfail(raises=SampleAppendError, strict=True)
def test_append_grayscale_second_generic_ds_unspecified_comp(
    local_ds_generator, deeplake_read_images
):
    "Append with htype=generic and sample_compression=unspecified."
    ds = local_ds_generator()
    _, gray, color = deeplake_read_images
    make_tensor_and_append(ds, "generic", "unspecified", [color, gray])


def test_append_two_grayscale(local_ds_generator, deeplake_read_images):
    "Append two deeplake.read grayscale images.  There should be no warning."
    ds = local_ds_generator()
    imgtype, gray, _ = deeplake_read_images
    make_tensor_and_append(ds, "image", imgtype, [gray, gray])
    assert len(ds.images) == 2
    assert ds.images._sample_shape_tensor.shape == (2, 3)
    for i in range(2):
        assert ds.images[i].numpy().shape == ds.images[i].shape
    assert list(ds.images.meta.min_shape) == list(gray.shape) + [1]
    assert list(ds.images.meta.max_shape) == list(gray.shape) + [1]


def test_append_many_grayscale(local_ds_generator, deeplake_read_images):
    "Append two deeplake.read grayscale images."
    ds = local_ds_generator()
    imgtype, gray, _ = deeplake_read_images
    make_tensor_and_append(ds, "image", imgtype, [gray, gray, gray, gray])
    assert len(ds.images) == 4
    assert ds.images._sample_shape_tensor.shape == (4, 3)
    for i in range(4):
        assert ds.images[i].numpy().shape == ds.images[i].shape
    assert list(ds.images.meta.min_shape) == list(gray.shape) + [1]
    assert list(ds.images.meta.max_shape) == list(gray.shape) + [1]
