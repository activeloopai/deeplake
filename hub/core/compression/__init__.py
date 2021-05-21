from .base import BaseNumCodec, BaseImgCodec
from .jpeg import JPEG
from .png import PNG
from .webp import WEBP
from .numpy import NUMPY
from .lz4 import LZ4
from .zstd import ZSTD

AVAILABLE_COMPRESSORS = [mod for mod in dir() if mod.isupper()]
