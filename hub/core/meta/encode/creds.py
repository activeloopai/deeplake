import json
from typing import Any
from hub.core.meta.encode.shape import ShapeEncoder
from hub.core.serialize import (
    deserialize_sequence_or_creds_encoder,
    serialize_sequence_or_creds_encoder,
)
from hub.core.storage.hub_memory_object import HubMemoryObject


class CredsEncoder(ShapeEncoder, HubMemoryObject):
    def register_samples(self, item: Any, num_samples: int):
        item = [item]
        return super().register_samples(item, num_samples)

    def __getitem__(
        self, local_sample_index: int, return_row_index: bool = False
    ) -> Any:
        return super().__getitem__(local_sample_index, return_row_index)

    def get_encoded_creds_key(self, local_sample_index: int):
        return self[local_sample_index][0]

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        if not buffer:
            return instance
        version, ids = deserialize_sequence_or_creds_encoder(buffer)
        if ids.nbytes:
            instance._encoded = ids
        instance.version = version
        instance.is_dirty = False
        return instance

    def tobytes(self) -> memoryview:
        return memoryview(
            serialize_sequence_or_creds_encoder(self.version, self._encoded)
        )
