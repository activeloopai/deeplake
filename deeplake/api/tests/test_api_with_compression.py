import os
import sys
from deeplake.constants import KB, MB
from deeplake.util.exceptions import (
    SampleCompressionError,
    TensorMetaMissingRequiredValue,
    TensorMetaMutuallyExclusiveKeysError,
    UnsupportedCompressionError,
    SampleHtypeMismatchError,
)
import pytest
from deeplake.core.tensor import Tensor
from deeplake.tests.common import TENSOR_KEY, assert_images_close
import numpy as np

import deeplake
from deeplake.core.dataset import Dataset


def _populate_compressed_samples(tensor: Tensor, cat_path, flower_path, count=1):
    for _ in range(count):
        tensor.append(deeplake.read(cat_path))
        tensor.append(deeplake.read(flower_path))
        tensor.append(np.ones((100, 100, 4), dtype="uint8"))
        tensor.append(
            np.ones((100, 100, 4), dtype=int).tolist()
        )  # test safe downcasting of python scalars

        tensor.extend(
            [
                deeplake.read(flower_path),
                deeplake.read(cat_path),
            ]
        )


def test_populate_compressed_samples(local_ds, cat_path, flower_path):
    images = local_ds.create_tensor(
        TENSOR_KEY,
        htype="image",
        sample_compression="png",
        max_chunk_size=2 * MB,
        tiling_threshold=1 * MB,
    )

    assert images.meta.dtype == "uint8"
    assert images.meta.sample_compression == "png"

    _populate_compressed_samples(images, cat_path, flower_path)

    expected_shapes = [
        (900, 900, 3),
        (513, 464, 4),
        (100, 100, 4),
        (100, 100, 4),
        (513, 464, 4),
        (900, 900, 3),
    ]

    assert len(images) == 6

    for img, exp_shape in zip(images, expected_shapes):
        arr = img.numpy()
        assert arr.shape == exp_shape
        assert arr.dtype == "uint8"

    assert images.shape == (6, None, None, None)
    assert images.shape_interval.lower == (6, 100, 100, 3)
    assert images.shape_interval.upper == (6, 900, 900, 4)


def test_iterate_compressed_samples(local_ds, cat_path, flower_path):
    images = local_ds.create_tensor(TENSOR_KEY, htype="image", sample_compression="png")

    assert images.meta.dtype == "uint8"
    assert images.meta.sample_compression == "png"

    _populate_compressed_samples(images, cat_path, flower_path)

    expected_shapes = [
        (900, 900, 3),
        (513, 464, 4),
        (100, 100, 4),
        (100, 100, 4),
        (513, 464, 4),
        (900, 900, 3),
    ]

    assert len(images) == len(expected_shapes)
    for image, expected_shape in zip(images, expected_shapes):
        x = image.numpy()

        assert (
            type(x) == np.ndarray
        ), "Check is necessary in case a `PIL` object is returned instead of an array."
        assert x.shape == expected_shape
        assert x.dtype == "uint8"


def test_uncompressed(local_ds):
    images = local_ds.create_tensor(TENSOR_KEY, sample_compression=None)

    images.append(np.ones((100, 100, 100)))
    images.extend(np.ones((3, 101, 2, 1)))
    local_ds.clear_cache()
    np.testing.assert_array_equal(images[0].numpy(), np.ones((100, 100, 100)))
    np.testing.assert_array_equal(images[1:4].numpy(), np.ones((3, 101, 2, 1)))


def test_byte_sample_compression(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("xyz", sample_compression="lz4")
        for i in range(10):
            ds.xyz.append(i * np.ones((100, 100, 100)))

    for i in range(10):
        np.testing.assert_array_equal(ds.xyz[i].numpy(), i * np.ones((100, 100, 100)))


@pytest.mark.xfail(raises=SampleCompressionError, strict=True)
@pytest.mark.parametrize(
    "bad_shape",
    [
        # raises OSError: cannot write mode LA as JPEG
        (100, 100, 2),
        # raises OSError: cannot write mode RGBA as JPE
        (100, 100, 4),
    ],
)
def test_jpeg_bad_shapes(memory_ds: Dataset, bad_shape):
    # jpeg allowed shapes:
    # ---------------------
    # (100) works!
    # (100,) works!
    # (100, 100) works!
    # (100, 100, 1) works!
    # (100, 100, 2) raises   | OSError: cannot write mode LA as JPEG
    # (100, 100, 3) works!
    # (100, 100, 4) raises   | OSError: cannot write mode RGBA as JPEG
    # (100, 100, 5) raises   | TypeError: Cannot handle this data type: (1, 1, 5), |u1
    # (100, 100, 100) raises | TypeError: Cannot handle this data type: (1, 1, 100), |u1

    tensor = memory_ds.create_tensor(TENSOR_KEY, sample_compression="jpeg")
    tensor.append(np.ones(bad_shape, dtype="uint8"))


def test_compression_aliases(memory_ds: Dataset):
    tensor = memory_ds.create_tensor("jpeg_tensor", sample_compression="jpeg")
    assert tensor.meta.sample_compression == "jpeg"

    tensor = memory_ds.create_tensor("jpg_tensor", sample_compression="jpg")
    assert tensor.meta.sample_compression == "jpeg"

    tensor = memory_ds.create_tensor("tiff_tensor", sample_compression="tiff")
    assert tensor.meta.sample_compression == "tiff"

    tensor = memory_ds.create_tensor("tif_tensor", sample_compression="tif")
    assert tensor.meta.sample_compression == "tiff"


@pytest.mark.xfail(raises=UnsupportedCompressionError, strict=True)
def test_unsupported_compression(memory_ds: Dataset):
    memory_ds.create_tensor(TENSOR_KEY, sample_compression="bad_compression")
    # TODO: same tests but with `dtype`


@pytest.mark.xfail(raises=TensorMetaMissingRequiredValue, strict=True)
def test_missing_sample_compression_for_image(memory_ds: Dataset):
    memory_ds.create_tensor("tensor", htype="image")


@pytest.mark.xfail(raises=TensorMetaMutuallyExclusiveKeysError, strict=True)
def test_sample_chunk_compression_mutually_exclusive(memory_ds: Dataset):
    memory_ds.create_tensor(
        "tensor", htype="image", sample_compression="png", chunk_compression="lz4"
    )


def test_chunkwise_compression(memory_ds, cat_path, flower_path):
    ds = memory_ds
    im_ct = 5
    chunk_size = 600 * KB
    with ds:
        images = ds.create_tensor(
            "images",
            htype="image",
            chunk_compression="jpg",
            max_chunk_size=chunk_size,
            tiling_threshold=chunk_size,
        )
        images.extend([deeplake.read(cat_path)] * im_ct)
        expected_arr = np.random.randint(0, 10, (500, 450, 3)).astype("uint8")
        images.append(expected_arr)
        images.extend([deeplake.read(cat_path)] * im_ct)
        expected_img = np.array(deeplake.read(cat_path))
    ds.clear_cache()
    for i, img in enumerate(images):
        if i == im_ct:
            assert_images_close(img.numpy(), expected_arr)
        else:
            assert_images_close(img.numpy(), expected_img)
    with ds:
        images = ds.create_tensor(
            "images2",
            htype="image",
            chunk_compression="png",
            max_chunk_size=chunk_size,
            tiling_threshold=chunk_size,
        )
        images.extend([deeplake.read(flower_path)] * im_ct)
        expected_arr = np.random.randint(0, 256, (200, 250, 4)).astype("uint8")
        images.append(expected_arr)
        images.extend([deeplake.read(flower_path)] * im_ct)
        expected_img = np.array(deeplake.read(flower_path))
    ds.clear_cache()
    for i, img in enumerate(images):
        if i == im_ct:
            assert_images_close(img.numpy(), expected_arr)
        else:
            assert_images_close(img.numpy(), expected_img)
    with ds:
        labels = ds.create_tensor(
            "labels",
            chunk_compression="lz4",
            max_chunk_size=chunk_size,
            tiling_threshold=chunk_size,
        )
        data = [
            np.random.randint(0, 256, (150, 150)).astype("uint8") for _ in range(20)
        ]
        labels.extend(data)
    ds.clear_cache()
    for row, label in zip(data, labels):
        np.testing.assert_array_equal(row, label.numpy())

    data = np.random.randint(0, 256, (5, 1500, 1500)).astype("uint8")
    with ds:
        ds.labels.extend(data)  # type: ignore
    ds.clear_cache()
    assert len(ds.labels) == 25
    for i in range(5):
        np.testing.assert_array_equal(data[i], ds.labels[20 + i].numpy())


def _decompress_audio_helper(path):
    import av  # type: ignore

    container = av.open(path)
    for frame in container.decode(audio=0):
        if not frame.is_corrupt:
            audio = frame.to_ndarray().astype("<f4")
            break

    for frame in container.decode(audio=0):
        audio = np.concatenate((audio, frame.to_ndarray().astype("<f4")), axis=1)

    return np.transpose(audio)


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
@pytest.mark.parametrize("compression", deeplake.compression.AUDIO_COMPRESSIONS)
def test_audio(local_ds, compression, audio_paths):
    path = audio_paths[compression]
    arr = _decompress_audio_helper(path)
    local_ds.create_tensor("audio", htype="audio", sample_compression=compression)
    with local_ds:
        for _ in range(10):
            local_ds.audio.append(deeplake.read(path))  # type: ignore
    for i in range(10):
        decompressed = local_ds.audio[i].numpy()
        np.testing.assert_array_equal(decompressed[: len(arr), :], arr)  # type: ignore


def test_exif(memory_ds, compressed_image_paths):
    ds = memory_ds
    with ds:
        ds.create_tensor("images", htype="image", sample_compression="jpeg")
        for path in compressed_image_paths["jpeg"]:
            ds.images.append(deeplake.read(path))
    for image in ds.images:
        assert isinstance(image.sample_info["exif"], dict), (
            type(image.sample_info["exif"]),
            path,
        )


def test_forced_htypes(
    memory_ds, grayscale_image_paths, color_image_paths, flower_path
):
    with memory_ds as ds:
        gray = ds.create_tensor("gray", htype="image.gray", sample_compression="jpeg")
        rgb = ds.create_tensor("rgb", htype="image.rgb", sample_compression="jpeg")

        gray.append(deeplake.read(grayscale_image_paths["jpeg"]))
        gray.append(deeplake.read(color_image_paths["jpeg"]))
        gray.append(deeplake.read(flower_path))
        gray.extend(np.ones((4, 10, 10, 3), dtype=np.uint8))
        gray.extend(
            [
                deeplake.read(color_image_paths["jpeg"]),
                np.ones((10, 10), dtype=np.uint8),
            ]
        )

        rgb.append(deeplake.read(grayscale_image_paths["jpeg"]))
        rgb.append(deeplake.read(color_image_paths["jpeg"]))
        rgb.append(deeplake.read(flower_path))
        rgb.extend(np.ones((4, 10, 10), dtype=np.uint8))
        rgb.extend(
            [
                deeplake.read(grayscale_image_paths["jpeg"]),
                np.ones((10, 10, 3), dtype=np.uint8),
            ]
        )

        gray_png = ds.create_tensor(
            "gray_png", htype="image.gray", sample_compression="png"
        )
        rgb_png = ds.create_tensor(
            "rgb_png", htype="image.rgb", sample_compression="png"
        )

        gray_png.append(deeplake.read(flower_path))
        gray_png.append(np.ones((10, 10, 4), dtype=np.uint8))

        rgb_png.append(deeplake.read(flower_path))
        rgb_png.append(np.ones((10, 10, 4), dtype=np.uint8))

        with pytest.raises(SampleHtypeMismatchError):
            rgb_png.append(1)

        with pytest.raises(TensorMetaMissingRequiredValue):
            ds.create_tensor("abc", htype="image.rgb")

        with pytest.raises(TensorMetaMissingRequiredValue):
            ds.create_tensor("abc", htype="image.gray")

    for sample in gray:
        assert len(sample.shape) == 2

    for sample in rgb:
        assert len(sample.shape) == 3

    for sample in gray_png:
        assert len(sample.shape) == 2

    for sample in rgb_png:
        assert len(sample.shape) == 3
