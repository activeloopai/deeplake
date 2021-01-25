"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import io

import numpy as np

from .base import Base


class Default(Base):
    def __init__(self):
        super().__init__()

    def encode(self, array: np.ndarray) -> bytes:
        with io.BytesIO() as f:
            np.save(f, array, allow_pickle=True)
            return f.getvalue()

    def decode(self, bytes_: bytes) -> np.ndarray:
        with io.BytesIO(bytes_) as f:
            return np.load(f, allow_pickle=True)
