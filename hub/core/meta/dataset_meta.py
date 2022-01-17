import json
from typing import Any, Dict
from hub.core.meta.meta import Meta


class DatasetMeta(Meta):
    def __init__(self):
        self.tensors = []
        self.groups = []
        self.has_agreement = False
        super().__init__()

    @property
    def nbytes(self):
        # TODO: can optimize this
        return len(self.tobytes())

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        if len(buffer) == 0:
            raise BufferError(
                "Unable to instantiate the object using frombuffer as the buffer was empty."
            )
        instance.__setstate__(json.loads(buffer))
        if not hasattr(instance, "has_agreement"):
            # None means we don't know whether it has agreement, this is for backwards compatibility
            instance.has_agreement = None
        return instance

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()
        d["tensors"] = self.tensors
        d["groups"] = self.groups
        d["has_agreement"] = self.has_agreement
        return d
