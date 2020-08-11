import zlib

from .zip import Zip


class Zlib(Zip):
    def __init__(self, compresslevel: int):
        super().__init__(zlib, compresslevel)
