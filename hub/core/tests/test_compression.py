from hub.tests.common import (
    assert_all_samples_have_expected_compression,
    get_actual_compression_from_buffer,
)
import numpy as np
import pytest
from hub.core.compression import compress_array, decompress_array


parametrize_compressions = pytest.mark.parametrize(
    "compression", ["jpeg", "png"]
)  # TODO: extend to be all pillow types we want to focus on


@parametrize_compressions
def test_array(compression):
    # TODO: check dtypes and no information loss
    array = np.zeros((100, 100, 3), dtype="uint8")  # TODO: handle non-uint8
    compressed_buffer = compress_array(array, compression)
    assert get_actual_compression_from_buffer(compressed_buffer) == compression
    decompressed_array = decompress_array(compressed_buffer)
    np.testing.assert_array_equal(array, decompressed_array)
