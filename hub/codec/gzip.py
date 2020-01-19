import gzip

from .zip import Zip

class Gzip(Zip):
    def __init__(self, compresslevel: float):
        compresslevel = int(min(9, round(compresslevel * 18)))
        super().__init__(gzip, compresslevel)
    