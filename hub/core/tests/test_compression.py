from hub.tests.common import get_actual_compression_from_buffer
import numpy as np
import pytest
import hub
from hub.core.compression import (
    compress_array,
    decompress_array,
    compress_multiple,
    decompress_multiple,
)
from hub.constants import SUPPORTED_COMPRESSIONS
from PIL import Image  # type: ignore


compressions = SUPPORTED_COMPRESSIONS[:]
compressions.remove(None)  # type: ignore
compressions.remove("wmf")  # driver has to be provided by user for wmf write support


@pytest.mark.parametrize("compression", compressions)
def test_array(compression, compressed_image_paths):
    # TODO: check dtypes and no information loss
    array = np.array(hub.read(compressed_image_paths[compression])) * False
    shape = array.shape
    compressed_buffer = compress_array(array, compression)
    assert get_actual_compression_from_buffer(compressed_buffer) == compression
    decompressed_array = decompress_array(compressed_buffer, shape=shape)
    np.testing.assert_array_equal(array, decompressed_array)


@pytest.mark.parametrize("compression", compressions)
def test_multi_array(compression, compressed_image_paths):
    img = Image.open(compressed_image_paths[compression])
    img2 = img.resize((img.size[0] // 2, img.size[1] // 2))
    img3 = img.resize((img.size[0] // 3, img.size[1] // 3))
    arrays = list(map(np.array, [img, img2, img3]))
    compressed_buffer = compress_multiple(arrays, compression)
    decompressed_arrays = decompress_multiple(compressed_buffer, [arr.shape for arr in arrays])
    for arr1, arr2 in zip(arrays, decompressed_arrays):
        if compression == "png":
            np.testing.assert_array_equal(arr1, arr2)
        else:
            assert arr1.shape == arr2.shape
