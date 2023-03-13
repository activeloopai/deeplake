from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject
from deeplake.core.serialize import (
    serialize_pad_encoder,
    deserialize_pad_encoder,
)
from deeplake.constants import ENCODING_DTYPE
import numpy as np
import deeplake


class PadEncoder(DeepLakeMemoryObject):
    def __init__(self):
        self.is_dirty = False
        self._encoded = np.zeros((0,), dtype=ENCODING_DTYPE)

    def add_padding(self, start_index: int, pad_length: int):
        self._encoded = np.concatenate([self._encoded, start_index, start_index + pad_length], axis=0)
        self.is_dirty = True

    def is_padded(self, global_sample_index: int):
        pass

    def tobytes(self) -> memoryview:
        return memoryview(
            serialize_pad_encoder(deeplake.__version__, self._encoded)
        )

    @classmethod
    def frombuffer(cls, buffer: bytes):
        isinstance = cls()
        if not buffer:
            return isinstance
        version, arr = deserialize_pad_encoder(buffer)
        if arr.nbytes:
            isinstance._encoded = arr
        isinstance.version = version
        return isinstance


    