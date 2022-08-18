import os
import sys
from hub.tests.common import get_actual_compression_from_buffer, assert_images_close
import numpy as np
import pytest
import hub
import lz4.frame  # type: ignore
from hub.core.compression import (
    compress_array,
    decompress_array,
    compress_multiple,
    decompress_multiple,
    verify_compressed_file,
    decompress_bytes,
)
from hub.compression import (
    get_compression_type,
    BYTE_COMPRESSION,
    IMAGE_COMPRESSION,
    IMAGE_COMPRESSIONS,
    BYTE_COMPRESSIONS,
    AUDIO_COMPRESSIONS,
    VIDEO_COMPRESSIONS,
    is_readonly_compression,
)
from hub.util.exceptions import CorruptedSampleError
from PIL import Image  # type: ignore


image_compressions = IMAGE_COMPRESSIONS[:]
image_compressions.remove("wmf")
image_compressions.remove("apng")
image_compressions.remove("dcm")

image_compressions = list(
    filter(lambda c: is_readonly_compression(c), image_compressions)
)


@pytest.mark.parametrize("compression", image_compressions + BYTE_COMPRESSIONS)
def test_array(compression, compressed_image_paths):
    # TODO: check dtypes and no information loss
    compression_type = get_compression_type(compression)
    if compression_type == BYTE_COMPRESSION:
        array = np.random.randint(0, 10, (32, 32))
    elif compression_type == IMAGE_COMPRESSION:
        array = np.array(hub.read(compressed_image_paths[compression][0]))
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


def test_lz4_bc():
    inp = np.random.random((100, 100)).tobytes()
    compressed = lz4.frame.compress(inp)
    decompressed = decompress_bytes(compressed, "lz4")
    assert decompressed == inp


def test_lz4_empty():
    assert decompress_bytes(b"", "lz4") == b""


@pytest.mark.parametrize("compression", AUDIO_COMPRESSIONS)
def test_audio(compression, audio_paths):
    path = audio_paths[compression]
    sample = hub.read(path)
    arr = np.array(sample)
    assert arr.dtype == "float32"
    with open(path, "rb") as f:
        assert sample.compressed_bytes(compression) == f.read()


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
@pytest.mark.parametrize("compression", VIDEO_COMPRESSIONS)
def test_video(compression, video_paths):
    for path in video_paths[compression]:
        sample = hub.read(path)
        arr = np.array(sample)
        assert arr.shape[-1] == 3
        assert arr.dtype == "uint8"
        with open(path, "rb") as f:
            assert sample.compressed_bytes(compression) == f.read()


def test_apng(memory_ds):
    ds = memory_ds

    arrays = {
        "binary": [
            np.random.randint(
                0, 256, (25, 50, np.random.randint(100, 200)), dtype=np.uint8
            )
            for _ in range(10)
        ],
        "rgb": [
            np.random.randint(
                0, 256, (np.random.randint(100, 200), 32, 64, 3), dtype=np.uint8
            )
            for _ in range(10)
        ],
        "rgba": [
            np.random.randint(
                0, 256, (np.random.randint(100, 200), 16, 32, 4), dtype=np.uint8
            )
            for _ in range(10)
        ],
    }
    for k, v in arrays.items():
        with ds:
            ds.create_tensor(k, htype="image", sample_compression="apng")
            ds[k].extend(v)
        for arr1, arr2 in zip(ds[k].numpy(aslist=True), v):
            np.testing.assert_array_equal(arr1, arr2)
