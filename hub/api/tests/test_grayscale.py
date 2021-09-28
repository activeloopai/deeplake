import pytest
import hub
from hub.util.exceptions import TensorInvalidSampleShapeError


@pytest.fixture(params=["jpeg"])
def image_gen(request, grayscale_image_paths, color_image_paths):
    gray_path = grayscale_image_paths[request.param]
    color_path = color_image_paths[request.param]
    yield request.param, hub.read(gray_path), hub.read(color_path)


def make_tensor_and_append(ds, imgtype, images):
    ds.create_tensor("images", htype="image", sample_compression=imgtype)
    for img in images:
        ds.images.append(img)


def make_tensor_and_extend(ds, imgtype, images):
    ds.create_tensor("images", htype="image", sample_compression=imgtype)
    ds.images.extend(images)


def test_append_grayscale_second(local_ds_generator, image_gen):
    ds = local_ds_generator()
    imgtype, gray, color = image_gen
    with pytest.warns(UserWarning, match=r"Converting to 3D"):
        make_tensor_and_append(ds, imgtype, [color, gray])
    assert len(ds.images) == 2
    assert ds.images.meta.min_shape[-1] == 1
    assert ds.images.meta.max_shape[-1] == 3


def test_append_grayscale_second_many(local_ds_generator, image_gen):
    ds = local_ds_generator()
    imgtype, gray, color = image_gen
    with pytest.warns(UserWarning, match=r"Converting to 3D"):
        make_tensor_and_append(ds, imgtype, [color, color, gray, color, gray, color])
    assert len(ds.images) == 6
    assert ds.images.meta.min_shape[-1] == 1
    assert ds.images.meta.max_shape[-1] == 3


def test_extend_grayscale_second(local_ds_generator, image_gen):
    ds = local_ds_generator()
    imgtype, gray, color = image_gen
    with pytest.warns(UserWarning, match=r"Converting to 3D"):
        make_tensor_and_extend(ds, imgtype, [color, gray])
    assert len(ds.images) == 2
    assert ds.images.meta.min_shape[-1] == 1
    assert ds.images.meta.max_shape[-1] == 3


@pytest.mark.xfail(raises=TensorInvalidSampleShapeError, strict=True)
def test_append_grayscale_first(local_ds_generator, image_gen):
    ds = local_ds_generator()
    imgtype, gray, color = image_gen
    make_tensor_and_append(ds, imgtype, [gray, color])


@pytest.mark.xfail(raises=TensorInvalidSampleShapeError, strict=True)
def test_append_grayscale_config_disable(local_ds_generator, image_gen, monkeypatch):
    ds = local_ds_generator()
    imgtype, gray, color = image_gen
    monkeypatch.setattr(hub.client.config, "CONVERT_GRAYSCALE_IMAGES_TO_3D", False)
    make_tensor_and_append(ds, imgtype, [color, gray])
