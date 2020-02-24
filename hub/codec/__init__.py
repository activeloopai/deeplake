from .base import Base
from .default import Default

from .zip import Zip
from .gzip import Gzip
from .lz4 import LZ4
from .zlib import Zlib

from .image import Image
from .jpeg import Jpeg
from .png import Png

def from_name(codec_name: str, compresslevel: float) -> Base:
    compresslevel = min(1, max(0, compresslevel))
    if codec_name == 'default':
        return Default()
    elif codec_name == 'gzip':
        return Gzip(compresslevel)
    elif codec_name == 'zlib':
        return Zlib(compresslevel)
    elif codec_name == 'lz4':
        return LZ4(compresslevel)
    elif codec_name == 'jpeg':
        return Jpeg()
    elif codec_name == 'png':
        return Png()
    else:
        raise Exception('Unknown Codec')