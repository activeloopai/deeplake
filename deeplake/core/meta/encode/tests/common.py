from hub.constants import ENCODING_DTYPE
import numpy as np


def assert_encoded(enc, expected_encoding):
    assert enc._encoded.dtype == ENCODING_DTYPE
    np.testing.assert_array_equal(enc._encoded, expected_encoding)
