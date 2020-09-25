from typing import Tuple, Iterable, Union
import numpy as np

from hub.features.features import Tensor
from hub.features.polygon import Polygon


class Mask(Tensor):
    def __init__(self, shape: Tuple[int, ...],
                 dtype = None):
        if not dtype:
            dtype = 'uint8'
        super(Mask, self).__init__(shape, dtype)

    def get_attr_dict(self):
        """Return class attributes
        """
        return self.__dict__
