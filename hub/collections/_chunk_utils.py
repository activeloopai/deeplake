import math

import numpy as np


def _logify_chunksize(chunksize):
    return 2 ** int(math.ceil(math.log2(max(chunksize, 1))))


def _tensor_chunksize(t: np.ndarray):
    sz = np.dtype(t.dtype).itemsize * np.prod(t.shape[1:])
    return _logify_chunksize((2 ** 24) / sz)
