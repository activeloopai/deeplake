import gzip

from .zip import Zip


class Gzip(Zip):
    def __init__(self, compresslevel: int):
        super().__init__(gzip, compresslevel)
