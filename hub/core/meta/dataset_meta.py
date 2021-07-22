from typing import Any, Dict
from hub.core.meta.meta import Meta


class DatasetMeta(Meta):
    def __init__(self):
        self.tensors = []

        super().__init__()

    @property
    def nbytes(self):
        # TODO: can optimize this
        return len(self.tobytes())

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()
        d["tensors"] = self.tensors
        return d
