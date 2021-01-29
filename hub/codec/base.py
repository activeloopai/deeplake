"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy


class Base:
    def encode(self, array: numpy.ndarray) -> bytes:
        raise NotImplementedError()

    def decode(self, bytes: bytes) -> numpy.ndarray:
        raise NotImplementedError()
