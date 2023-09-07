import os
import sys
from deeplake.tests.common import (
    get_actual_compression_from_buffer,
    assert_images_close,
)
import numpy as np
import pytest
import deeplake
import lz4.frame  # type: ignore
from deeplake.core.compression import (
    compress_array,
    decompress_array,
    compress_multiple,
    decompress_multiple,
    verify_compressed_file,
    decompress_bytes,
)
from deeplake.compression import (
    get_compression_type,
    BYTE_COMPRESSION,
    IMAGE_COMPRESSION,
    IMAGE_COMPRESSIONS,
    BYTE_COMPRESSIONS,
    AUDIO_COMPRESSIONS,
    VIDEO_COMPRESSIONS,
    is_readonly_compression,
    SUPPORTED_COMPRESSIONS,
    POINT_CLOUD_COMPRESSIONS,
)
from deeplake.util.exceptions import CorruptedSampleError
from PIL import Image  # type: ignore


image_compressions = IMAGE_COMPRESSIONS[:]
image_compressions.remove("wmf")
# image_compressions.remove("apng")
image_compressions.remove("dcm")
image_compressions.remove("fli")
image_compressions.remove("mpo")

image_compressions = list(
    filter(lambda c: is_readonly_compression(c), image_compressions)
)


@pytest.mark.parametrize("compression", image_compressions + BYTE_COMPRESSIONS)
@pytest.mark.slow
def test_array(compression, compressed_image_paths):
    # TODO: check dtypes and no information loss
    compression_type = get_compression_type(compression)
    if compression_type == BYTE_COMPRESSION:
        array = np.random.randint(0, 10, (32, 32))
    elif compression_type == IMAGE_COMPRESSION:
        array = np.array(deeplake.read(compressed_image_paths[compression][0]))
    shape = array.shape
    compressed_buffer = compress_array(array, compression)
    if compression_type == BYTE_COMPRESSION:
        decompressed_array = decompress_array(
            compressed_buffer, shape=shape, dtype=array.dtype, compression=compression
        )
    else:
        assert get_actual_compression_from_buffer(compressed_buffer) == compression
        decompressed_array = decompress_array(compressed_buffer, shape=shape)
    if compression == "png" or compression_type == BYTE_COMPRESSION:
        np.testing.assert_array_equal(array, decompressed_array)
    else:
        assert_images_close(array, decompressed_array)


@pytest.mark.parametrize("compression", image_compressions + BYTE_COMPRESSIONS)
@pytest.mark.slow
def test_multi_array(compression, compressed_image_paths):
    compression_type = get_compression_type(compression)
    if compression_type == IMAGE_COMPRESSION:
        img = Image.open(compressed_image_paths[compression][0])
        img2 = img.resize((img.size[0] // 2, img.size[1] // 2))
        img3 = img.resize((img.size[0] // 3, img.size[1] // 3))
        arrays = list(map(np.array, [img, img2, img3]))
        compressed_buffer = compress_multiple(arrays, compression)
        decompressed_arrays = decompress_multiple(
            compressed_buffer, [arr.shape for arr in arrays]
        )
    elif compression_type == BYTE_COMPRESSION:
        arrays = [np.random.randint(0, 10, (32, 32)) for _ in range(3)]
        compressed_buffer = compress_multiple(arrays, compression)
        decompressed_arrays = decompress_multiple(
            compressed_buffer, [(32, 32)] * 3, arrays[0].dtype, compression
        )

    for arr1, arr2 in zip(arrays, decompressed_arrays):
        if compression == "png" or compression_type == BYTE_COMPRESSION:
            np.testing.assert_array_equal(arr1, arr2)
        else:
            assert_images_close(arr1, arr2)


@pytest.mark.parametrize("compression", image_compressions)
@pytest.mark.slow
def test_verify(compression, compressed_image_paths, corrupt_image_paths):
    for path in compressed_image_paths[compression]:
        sample = deeplake.read(path)
        sample_loaded = deeplake.read(path)
        sample_loaded.compressed_bytes(compression)
        sample_verified_and_loaded = deeplake.read(path, verify=True)
        sample_verified_and_loaded.compressed_bytes(compression)
        pil_image_shape = np.array(Image.open(path)).shape
        assert (
            sample.shape
            == sample_loaded.shape
            == sample_verified_and_loaded.shape
            == pil_image_shape
        ), (
            sample.shape,
            sample_loaded.shape,
            sample_verified_and_loaded.shape,
            pil_image_shape,
        )
        verify_compressed_file(path, compression)
        with open(path, "rb") as f:
            verify_compressed_file(f, compression)
    if compression in corrupt_image_paths:
        path = corrupt_image_paths[compression]
        sample = deeplake.read(path)
        sample.compressed_bytes(compression)
        Image.open(path)
        with pytest.raises(CorruptedSampleError):
            sample = deeplake.read(path, verify=True)
            sample.compressed_bytes(compression)
        with pytest.raises(CorruptedSampleError):
            verify_compressed_file(path, compression)
        with pytest.raises(CorruptedSampleError):
            with open(path, "rb") as f:
                verify_compressed_file(f, compression)
        with pytest.raises(CorruptedSampleError):
            with open(path, "rb") as f:
                verify_compressed_file(f.read(), compression)


def test_lz4_bc():
    inp = np.random.random((100, 100)).tobytes()
    compressed = lz4.frame.compress(inp)
    decompressed = decompress_bytes(compressed, "lz4")
    assert decompressed == inp


def test_lz4_empty():
    assert decompress_bytes(b"", "lz4") == b""


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
@pytest.mark.parametrize("compression", AUDIO_COMPRESSIONS)
def test_audio(compression, audio_paths):
    path = audio_paths[compression]
    sample = deeplake.read(path)
    arr = np.array(sample)
    assert arr.dtype == "float32"
    with open(path, "rb") as f:
        assert sample.compressed_bytes(compression) == f.read()


@pytest.mark.slow
@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
@pytest.mark.parametrize("compression", VIDEO_COMPRESSIONS)
def test_video(compression, video_paths):
    for path in video_paths[compression]:
        sample = deeplake.read(path)
        arr = np.array(sample)
        assert arr.shape[-1] == 3
        assert arr.dtype == "uint8"
        with open(path, "rb") as f:
            assert sample.compressed_bytes(compression) == f.read()


@pytest.mark.parametrize("compression", POINT_CLOUD_COMPRESSIONS)
def test_point_cloud(point_cloud_paths, compression):
    path = point_cloud_paths[compression]
    sample = deeplake.read(path)
    arr = np.array(sample)
    if compression == "las":
        assert len(arr[0]) == 18


# def test_apng(memory_ds):
#     ds = memory_ds

#     arrays = {
#         "binary": [
#             np.random.randint(
#                 0, 256, (25, 50, np.random.randint(100, 200)), dtype=np.uint8
#             )
#             for _ in range(10)
#         ],
#         "rgb": [
#             np.random.randint(
#                 0, 256, (np.random.randint(100, 200), 32, 64, 3), dtype=np.uint8
#             )
#             for _ in range(10)
#         ],
#         "rgba": [
#             np.random.randint(
#                 0, 256, (np.random.randint(100, 200), 16, 32, 4), dtype=np.uint8
#             )
#             for _ in range(10)
#         ],
#     }
#     for k, v in arrays.items():
#         with ds:
#             ds.create_tensor(k, htype="image", sample_compression="apng")
#             ds[k].extend(v)
#         for arr1, arr2 in zip(ds[k].numpy(aslist=True), v):
#             np.testing.assert_array_equal(arr1, arr2)
