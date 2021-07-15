from hub.core.storage.cachable import Cachable


class Info(Cachable):
    def __init__(self):
        """Contains **optional** key/values that datasets/tensors use for human-readability.
        See the `Meta` class for required key/values for datasets/tensors.
        """

        pass

    def as_dict(self) -> dict:
        raise NotImplementedError
