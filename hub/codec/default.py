import numpy
import pickle

from .base import Base

class Default(Base):
    def __init__(self):
        super().__init__()

    def encode(self, array: numpy.ndarray) -> bytes:
        info = {}
        info['shape'] = array.shape
        info['dtype'] = array.dtype
        info['data'] = array.tobytes()
        return pickle.dumps(info)

    def decode(self, bytes: bytes) -> numpy.ndarray:
        info = pickle.loads(bytes)
        arr = numpy.frombuffer(bytearray(info['data']), dtype=info['dtype']).reshape(info['shape'])
        return arr