"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from .base import Base
from .default import Default

from .gzip import Gzip
from .lz4 import LZ4
from .zlib import Zlib

from .jpeg import Jpeg
from .png import Png


def from_name(codec_name: str) -> Base:
    if codec_name is None:
        return Default()
    compresslevel = None
    if ":" in codec_name:
        comp = codec_name.split(":")
        assert (
            len(comp) == 2
        ), f"{codec_name} is invalid compress format, should be [type] or [type:level]"
        codec_name = comp[0]
        compresslevel = int(comp[1])
    if codec_name == "default":
        return Default()
    elif codec_name == "gzip":
        return Gzip(compresslevel or 9)
    elif codec_name == "zlib":
        return Zlib(compresslevel or -1)
    elif codec_name == "lz4":
        return LZ4(compresslevel or 1)
    elif codec_name == "jpeg":
        return Jpeg()
    elif codec_name == "png":
        return Png()
    else:
        raise Exception("Unknown Codec")
