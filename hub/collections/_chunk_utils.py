"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import math

import numpy as np


def _logify_chunksize(chunksize):
    return 2 ** int(math.ceil(math.log2(max(chunksize, 1))))


def _tensor_chunksize(t: np.ndarray):
    sz = np.dtype(t.dtype).itemsize * np.prod(t.shape[1:])
    return _logify_chunksize((2 ** 24) / sz)
