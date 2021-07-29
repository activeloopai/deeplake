from hub.tests.common import get_actual_compression_from_buffer
import numpy as np
import pytest
from hub.core.compression import compress_array, decompress_array
from hub.constants import SUPPORTED_COMPRESSIONS
from PIL import Image


compressions = SUPPORTED_COMPRESSIONS[:]
compressions.remove(None)
compressions.remove("wmf")  # driver has to be provided by user for write support


@pytest.mark.parametrize("compression", compressions)
def test_array(compression, compressed_image_paths):
    # TODO: check dtypes and no information loss
    array = np.array(Image.open(compressed_image_paths[compression])) * False
    shape = array.shape
    compressed_buffer = compress_array(array, compression)
    assert get_actual_compression_from_buffer(compressed_buffer) == compression
    decompressed_array = decompress_array(compressed_buffer, shape=shape)
    np.testing.assert_array_equal(array, decompressed_array)
