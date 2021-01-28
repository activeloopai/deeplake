"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import gzip

from .zip import Zip


class Gzip(Zip):
    def __init__(self, compresslevel: int):
        super().__init__(gzip, compresslevel)
