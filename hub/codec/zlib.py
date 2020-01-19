import zlib

from .zip import Zip

class Zlib(Zip):
    def __init__(self, compresslevel: float):
        compresslevel = int(round(6 + (compresslevel - 0.5) * 3)) if compresslevel >= 0.5 else int(round(compresslevel * 12))
        super().__init__(zlib, compresslevel)