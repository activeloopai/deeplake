from typing import Any, Optional
from hub.constants import ENCODING_DTYPE, UNSHARDED_ENCODER_FILENAME
from hub.core.meta.encode.shape import ShapeEncoder
from hub.core.serialize import deserialize_generic_encoder, serialize_generic_encoder
from hub.core.storage.hub_memory_object import HubMemoryObject
import numpy as np



class EncoderMap(ShapeEncoder, HubMemoryObject):
    def __init__(self):
        self.is_dirty = False
        super().__init__(dtype=np.uint64)

    def generate_encoder_id(self):
        if len(self.array) == 0:
            return 0
        return self.array[-1][0] + 1

    def add_encoder(self):
        self.is_dirty = True
        encoder_id = self.generate_encoder_id()
        offset = self.num_samples
        self.register_samples(encoder_id, offset)
        return self.get_last_encoder_name(), self.get_encoder_dtype(encoder_id)

    def register_samples(self, item: Any, num_samples: int, row: Optional[int] = None):
        self.is_dirty = True
        return super().register_samples(item, num_samples)

    def get_encoder_name(self, global_sample_index: int):
        encoder_id = self[global_sample_index][0]
        if encoder_id == 0:
            return UNSHARDED_ENCODER_FILENAME
        return f"shard_{encoder_id}"

    def get_last_encoder_name(self):
        return self.get_encoder_name(self.num_samples - 1)

    def get_encoder_dtype(self, encoder_id):
        if encoder_id == 0:
            return "uint64"
        else:
            return "uint128"


    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        if not buffer:
            return instance
        version, ids = deserialize_generic_encoder(buffer, "map", np.uint64)
        if ids.nbytes:
            instance._encoded = ids
        instance.version = version
        instance.is_dirty = False
        return instance

    def tobytes(self) -> memoryview:
        return memoryview(
            serialize_generic_encoder(self.version, self._encoded)
        )
        
    @property
    def nbytes(self):
        return len(self.tobytes())