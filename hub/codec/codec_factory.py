from typing import Optional
from .codec import Codec
from .default_codec import DefaultCodec

from .gzip_codec import GzipCodec
from .zlib_codec import ZlibCodec
from .lz4_codec import LZ4Codec

from .jpeg_codec import JpegCodec
from .png_codec import PngCodec


class CodecFactory():
    @staticmethod
    def create(codec_name: str, compresslevel: float) -> Codec:
        compresslevel = min(1, max(0, compresslevel))
        if codec_name == 'default':
            return DefaultCodec()
        elif codec_name == 'gzip':
            return GzipCodec(compresslevel)
        elif codec_name == 'zlib':
            return ZlibCodec(compresslevel)
        elif codec_name == 'lz4':
            return LZ4Codec(compresslevel)
        elif codec_name == 'jpeg':
            return JpegCodec()
        elif codec_name == 'png':
            return PngCodec()
        else:
            raise Exception('Unknown Codec')

