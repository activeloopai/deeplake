from typing import *
import os, sys, io, time, random, traceback, json

import numpy as np
import pickle as msgpack

from .base import Base

class Zip(Base):
    def __init__(self, compressor, compresslevel: int):
        self._compressor = compressor
        self._compresslevel = compresslevel

    def encode(self, array: np.ndarray) -> bytes:
        info = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'data': self._compressor.compress(array.tobytes(), self._compresslevel),
        }
        return msgpack.dumps(info)
    
    def decode(self, content: bytes) -> np.ndarray:
        info = msgpack.loads(content)
        data = self._compressor.decompress(info['data'])
        return np.frombuffer(bytearray(data), dtype=info['dtype'])\
                                            .reshape(info['shape'])

    