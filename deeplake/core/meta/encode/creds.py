import json
from typing import Any, Optional
from deeplake.core.meta.encode.shape import ShapeEncoder
from deeplake.core.serialize import (
    deserialize_sequence_or_creds_encoder,
    serialize_sequence_or_creds_encoder,
)
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject


class CredsEncoder(ShapeEncoder, DeepLakeMemoryObject):
    def __init__(self):
        self.is_dirty = False
        super().__init__()

    def register_samples(self, item: Any, num_samples: int, row: Optional[int] = None):
        self.is_dirty = True
        return super().register_samples(item, num_samples)

    def __setitem__(self, local_sample_index: int, item: Any):
        self.is_dirty = True
        return super().__setitem__(local_sample_index, item)

    def get_encoded_creds_key(self, local_sample_index: int):
        return self[local_sample_index][0]

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        if not buffer:
            return instance
        version, ids = deserialize_sequence_or_creds_encoder(buffer, "creds")
        if ids.nbytes:
            instance._encoded = ids
        instance.version = version
        instance.is_dirty = False
        return instance

    def tobytes(self) -> memoryview:
        return memoryview(
            serialize_sequence_or_creds_encoder(self.version, self._encoded)
        )

    @property
    def nbytes(self):
        return len(self.tobytes())

    def pop(self, index):
        self.is_dirty = True
        super().pop(index)
