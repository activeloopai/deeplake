from typing import Any, Dict
import deeplake
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject


class Meta(DeepLakeMemoryObject):
    """Contains **required** key/values that datasets/tensors use to function.
    See the ``Info`` class for optional key/values for datasets/tensors.
    """

    def __init__(self):
        super().__init__()
        self.version = deeplake.__version__

    def __getstate__(self) -> Dict[str, Any]:
        return {"version": self.version}
