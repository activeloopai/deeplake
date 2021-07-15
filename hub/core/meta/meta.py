import hub
from hub.core.storage.cachable import Cachable


class Meta(Cachable):
    def __init__(self):
        """Contains **required** key/values that datasets/tensors use to function.
        See the `Info` class for optional key/values for datasets/tensors.
        """

        self.version = hub.__version__

    def as_dict(self) -> dict:
        return {"version": self.version}
