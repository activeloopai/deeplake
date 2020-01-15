import numpy
import pickle

from .codec import Codec

class ZipCodec(Codec):
    def __init__(self, compressor, compresslevel: int):
        self._compressor = compressor
        self._compresslevel = compresslevel

    def encode(self, array: numpy.ndarray) -> bytes:
        info = {}
        info['shape'] = array.shape
        info['dtype'] = array.dtype
        info['data'] = self._compressor.compress(array.tobytes(), self._compresslevel)
        return pickle.dumps(info)
    
    def decode(self, content: bytes) -> numpy.ndarray:
        info = pickle.loads(content)
        data = self._compressor.decompress(info['data'])
        return numpy.frombuffer(data, dtype=info['dtype']).reshape(info['shape'])

    