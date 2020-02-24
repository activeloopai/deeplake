import lz4.frame

from .zip import Zip

class LZ4(Zip):
    def __init__(self, compresslevel: float):
        compresslevel =  int(max(0, round(compresslevel * 32 - 16)))
        super().__init__(lz4.frame, compresslevel)
        
