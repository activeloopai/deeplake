import gzip

from .zip_codec import ZipCodec

class GzipCodec(ZipCodec):
    def __init__(self, compresslevel: float):
        compresslevel = int(min(9, round(compresslevel * 18)))
        super().__init__(gzip, compresslevel)
    