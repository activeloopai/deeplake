from hub.tests.common import get_actual_compression_from_buffer
import numpy as np
import pytest
import hub
from hub.core.compression import compress_array, decompress_array
from hub.util.exceptions import CorruptedSampleError
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
def test_verify(compression, compressed_image_paths, corrupt_image_paths):
    path = compressed_image_paths[compression]
    sample = hub.read(path, verify=True)
    sample.compressed_bytes(compression)
    if compression in corrupt_image_paths:
        path = corrupt_image_paths[compression]
        sample = hub.read(path)
        sample.compressed_bytes(compression)
        Image.open(path)
        with pytest.raises(CorruptedSampleError):
            sample = hub.read(path, verify=True)
            sample.compressed_bytes(compression)
