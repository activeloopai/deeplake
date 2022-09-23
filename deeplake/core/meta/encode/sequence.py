from hub.core.meta.encode.byte_positions import BytePositionsEncoder
from hub.core.storage.hub_memory_object import HubMemoryObject
from hub.core.serialize import (
    serialize_sequence_or_creds_encoder,
    deserialize_sequence_or_creds_encoder,
)


class SequenceEncoder(BytePositionsEncoder, HubMemoryObject):
    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        if not buffer:
            return instance
        version, ids = deserialize_sequence_or_creds_encoder(buffer, "seq")
        if ids.nbytes:
            instance._encoded = ids
        instance.version = version
        instance.is_dirty = False
        return instance

    def tobytes(self) -> memoryview:
        return memoryview(
            serialize_sequence_or_creds_encoder(self.version, self._encoded)
        )

    def pop(self, index):
        self.is_dirty = True
        super().pop(index)
