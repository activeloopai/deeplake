import lz4.frame

from .zip import Zip


class LZ4(Zip):
    def __init__(self, compresslevel: int):
        super().__init__(lz4.frame, compresslevel)
