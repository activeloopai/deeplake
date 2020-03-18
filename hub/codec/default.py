from typing import *
import os, sys, io, time, random, traceback, json

import numpy as np
import pickle as msgpack


from .base import Base

class Default(Base):
    def __init__(self):
        super().__init__()

    def encode(self, array: np.ndarray) -> bytes:
        info = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'data': array.tobytes(),
        }
        return msgpack.dumps(info)

    def decode(self, bytes: bytes) -> np.ndarray:
        info = msgpack.loads(bytes)
        return np.frombuffer(bytearray(info['data']), dtype=info['dtype'])\
                                .reshape(info['shape'])