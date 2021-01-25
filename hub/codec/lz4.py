"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import lz4.frame

from .zip import Zip


class LZ4(Zip):
    def __init__(self, compresslevel: int):
        super().__init__(lz4.frame, compresslevel)
