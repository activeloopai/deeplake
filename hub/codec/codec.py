import numpy
import pickle

class Codec():
    def encode(self, array: numpy.ndarray) -> bytes:
        raise NotImplementedError()

    def decode(self, bytes: bytes) -> numpy.ndarray:
        raise NotImplementedError()