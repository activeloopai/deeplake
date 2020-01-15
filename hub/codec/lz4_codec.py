import lz4.frame

from .zip_codec import ZipCodec

class LZ4Codec(ZipCodec):
    def __init__(self, compresslevel: float):
        compresslevel =  int(max(0, round(compresslevel * 32 - 16)))
        super().__init__(lz4.frame, compresslevel)
        
