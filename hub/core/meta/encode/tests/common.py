import numpy as np


def assert_encoded(enc, expected_encoding):
    np.testing.assert_array_equal(enc._encoded, expected_encoding)
