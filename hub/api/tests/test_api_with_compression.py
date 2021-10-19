from hub.compression import get_compression_type
from hub.util.exceptions import (
    SampleCompressionError,
    TensorMetaMissingRequiredValue,
    TensorMetaMutuallyExclusiveKeysError,
    UnsupportedCompressionError,
)
import pytest
from hub.core.tensor import Tensor
from hub.tests.common import TENSOR_KEY, assert_images_close
from hub.tests.dataset_fixtures import enabled_datasets
import numpy as np

import hub
from hub.core.dataset import Dataset
from miniaudio import mp3_read_file_f32, flac_read_file_f32, wav_read_file_f32  # type: ignore


def _populate_compressed_samples(tensor: Tensor, cat_path, flower_path, count=1):
    for _ in range(count):
        tensor.append(hub.read(cat_path))
        tensor.append(hub.read(flower_path))
        tensor.append(np.ones((100, 100, 4), dtype="uint8"))
        tensor.append(
            np.ones((100, 100, 4), dtype=int).tolist()
        )  # test safe downcasting of python scalars

        tensor.extend(
            [
                hub.read(flower_path),
                hub.read(cat_path),
            ]
        )


@enabled_datasets
def test_populate_compressed_samples(ds: Dataset, cat_path, flower_path):
    images = ds.create_tensor(TENSOR_KEY, htype="image", sample_compression="png")

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


@enabled_datasets
def test_iterate_compressed_samples(ds: Dataset, cat_path, flower_path):
    images = ds.create_tensor(TENSOR_KEY, htype="image", sample_compression="png")

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


@enabled_datasets
def test_uncompressed(ds: Dataset):
    images = ds.create_tensor(TENSOR_KEY, sample_compression=None)

    images.append(np.ones((100, 100, 100)))
    images.extend(np.ones((3, 101, 2, 1)))
    ds.clear_cache()
    np.testing.assert_array_equal(images[0].numpy(), np.ones((100, 100, 100)))
    np.testing.assert_array_equal(images[1:4].numpy(), np.ones((3, 101, 2, 1)))


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


@enabled_datasets
def test_chunkwise_compression(ds: Dataset, cat_path, flower_path):
    images = ds.create_tensor("images", htype="image", chunk_compression="jpg")
    images.append(hub.read(cat_path))
    images.append(hub.read(cat_path))
    expected_arr = np.random.randint(0, 10, (500, 450, 3)).astype("uint8")
    images.append(expected_arr)
    images.append(hub.read(cat_path))
    images.append(hub.read(cat_path))
    expected_img = np.array(hub.read(cat_path))
    assert_images_close(images[2].numpy(), expected_arr)
    for img in images[[0, 1, 3, 4]]:
        assert_images_close(img.numpy(), expected_img)

    images = ds.create_tensor("images2", htype="image", chunk_compression="png")
    images.append(hub.read(flower_path))
    images.append(hub.read(flower_path))
    expected_arr = np.random.randint(0, 256, (500, 450, 4)).astype("uint8")
    images.append(expected_arr)
    images.append(hub.read(flower_path))
    images.append(hub.read(flower_path))
    expected_img = np.array(hub.read(flower_path))

    np.testing.assert_array_equal(images[2].numpy(), expected_arr)
    for img in images[[0, 1, 3, 4]]:
        np.testing.assert_array_equal(img, expected_img)

    labels = ds.create_tensor("labels", chunk_compression="lz4")
    data = [[0] * 50, [1, 2, 3] * 100, [4, 5, 6] * 200, [7, 8, 9] * 300]
    labels.extend(data)
    for row, label in zip(data, labels):
        np.testing.assert_array_equal(row, label.numpy())


@enabled_datasets
@pytest.mark.parametrize("compression", hub.compression.AUDIO_COMPRESSIONS)
def test_audio(ds: Dataset, compression, audio_paths):
    path = audio_paths[compression]
    if path.endswith(".mp3"):
        audio = mp3_read_file_f32(path)
    elif path.endswith(".flac"):
        audio = flac_read_file_f32(path)
    elif path.endswith(".wav"):
        audio = wav_read_file_f32(path)
    arr = np.frombuffer(audio.samples, dtype=np.float32).reshape(
        audio.num_frames, audio.nchannels
    )
    ds.create_tensor("audio", htype="audio", sample_compression=compression)
    with ds:
        for _ in range(10):
            ds.audio.append(hub.read(path))  # type: ignore
    for i in range(10):
        np.testing.assert_array_equal(ds.audio[i].numpy(), arr)  # type: ignore
