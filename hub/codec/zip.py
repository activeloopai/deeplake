import io

import numpy as np

from .base import Base


class Zip(Base):
    def __init__(self, compressor, compresslevel: int):
        self._compressor = compressor
        self._compresslevel = compresslevel

    def encode(self, array: np.ndarray) -> bytes:
        with io.BytesIO() as f:
            np.save(f, array, allow_pickle=True)
            return self._compressor.compress(f.getvalue(), self._compresslevel)

    def decode(self, content: bytes) -> np.ndarray:
        with io.BytesIO(self._compressor.decompress(content)) as f:
            return np.load(f, allow_pickle=True)
