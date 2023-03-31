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
        self._buffer = []
        self.version = deeplake.__version__

    def _flush(self):
        if self._buffer:
            self._encoded = np.concatenate(
                [self._encoded, np.array(self._buffer, dtype=ENCODING_DTYPE)], axis=0
            )
            self._buffer = []

    def add_padding(self, start_index: int, pad_length: int) -> None:
        self._buffer += [start_index, start_index + pad_length]
        self.is_dirty = True

    def _is_padded(self, global_sample_index: int, idx: int) -> bool:
        if idx >= len(self._encoded):
            return False
        m = idx % 2
        edge = self._encoded[idx]
        if m == 0:
            return edge == global_sample_index
        else:
            return global_sample_index < edge

    def is_padded(self, global_sample_index: int) -> bool:
        self._flush()
        idx = np.searchsorted(self._encoded, global_sample_index)
        return self._is_padded(global_sample_index, idx)  # type: ignore

    def tobytes(self) -> memoryview:
        self._flush()
        return memoryview(serialize_pad_encoder(self.version, self._encoded))

    def _unpad(self, global_sample_index: int, idx: int) -> None:
        if (
            global_sample_index == self._encoded[idx]
            and self._encoded[idx] + 1 == self._encoded[idx + 1]
        ):
            self._encoded = np.concatenate(
                [self._encoded[:idx], self._encoded[idx + 2 :]],
                axis=0,
            )
        elif global_sample_index + 1 == self._encoded[idx]:
            self._encoded = np.concatenate(
                [
                    self._encoded[:idx],
                    np.array(
                        [global_sample_index, global_sample_index + 1],
                        dtype=ENCODING_DTYPE,
                    ),
                    self._encoded[idx + 1 :],
                ],
                axis=0,
            )
        else:
            self._encoded = np.concatenate(
                [
                    self._encoded[:idx],
                    np.array(
                        [global_sample_index, global_sample_index + 1],
                        dtype=ENCODING_DTYPE,
                    ),
                    self._encoded[idx:],
                ],
                axis=0,
                dtype=ENCODING_DTYPE,
            )

    def unpad(self, global_sample_index: int) -> None:
        self._flush()
        idx = np.searchsorted(self._encoded, global_sample_index)
        if self._is_padded(global_sample_index, idx):  # type: ignore
            self._unpad(global_sample_index, idx)  # type: ignore

    def pop(self, global_sample_index: int) -> None:
        self._flush()
        idx = np.searchsorted(self._encoded, global_sample_index)
        is_padded = self._is_padded(global_sample_index, idx)  # type: ignore
        if is_padded:
            self._unpad(global_sample_index, idx)  # type: ignore
            self._encoded[idx + 1] -= 1
        else:
            self._encoded[idx:] -= 1  # type: ignore

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        if not buffer:
            return instance
        version, arr = deserialize_pad_encoder(buffer)
        if arr.nbytes:
            instance._encoded = arr
        instance.version = version
        return instance

    @property
    def nbytes(self):
        return self._encoded.dtype.itemsize * (self._encoded.size * len(self._buffer))

    @property
    def array(self):
        self._flush()
        return self._encoded
