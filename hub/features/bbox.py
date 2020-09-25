from typing import Tuple

from hub.features.features import Tensor


class BBox(Tensor):
    def __init__(self, dtype = None):
        if not dtype:
            dtype = 'float32'
        super(BBox, self).__init__(shape=(4,), dtype=dtype)

    def get_attr_dict(self):
        """Return class attributes
        """
        return self.__dict__ 
