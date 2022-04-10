import json
from typing import Any
from hub.core.meta.encode.shape import ShapeEncoder
from hub.core.serialize import (
    deserialize_sequence_or_creds_encoder,
    serialize_sequence_or_creds_encoder,
)
from hub.core.storage.hub_memory_object import HubMemoryObject


class CredsEncoder(ShapeEncoder, HubMemoryObject):
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
