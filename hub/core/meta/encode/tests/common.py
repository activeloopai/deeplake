from numpy import testing as np_testing

from hub.constants import ENCODING_DTYPE


def assert_encoded(enc, expected_encoding):
    assert enc._encoded.dtype == ENCODING_DTYPE
    np_testing.assert_array_equal(enc._encoded, expected_encoding)
