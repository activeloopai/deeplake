from typing import Tuple
# import numpy as np

from hub.features.features import Tensor
# from hub.features.polygon import Polygon


class Mask(Tensor):
    """`FeatureConnector` for mask
    """
    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        dtype=None,
        max_shape: Tuple[int, ...] = None,
        chunks=True
    ):
        """Constructs a Mask FeatureConnector.

        shape: tuple of ints or None: (height, width, 1)
        dtype: dtype of mask array. Default: `uint8`

        Example:
        ```python
        mask_tensor = Mask(shape=(300, 300, 1))
        ```
        """
        if not dtype:
            dtype = 'uint8'
        super(Mask, self).__init__(shape, dtype, max_shape=max_shape, chunks=chunks)

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
