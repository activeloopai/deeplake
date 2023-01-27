import numpy as np


def assert_array_equal(x, y, *args, **kwargs):
    if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
        assert len(x) == len(y)
        for xi, yi in zip(x, y):
            return np.testing.assert_array_equal(xi, yi, *args, **kwargs)
    else:
        return np.testing.assert_array_equal(x, y, *args, **kwargs)
