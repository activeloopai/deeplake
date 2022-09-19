from typing import Any, Dict
import hub
from hub.core.storage.hub_memory_object import HubMemoryObject


class Meta(HubMemoryObject):
    """Contains **required** key/values that datasets/tensors use to function.
    See the ``Info`` class for optional key/values for datasets/tensors.
    """

    def __init__(self):
        super().__init__()
        self.version = hub.__version__

    def __getstate__(self) -> Dict[str, Any]:
        return {"version": self.version}
