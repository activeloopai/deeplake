from typing import Tuple

# import numpy as np

from hub.features.features import Tensor

# from hub.features.polygon import Polygon


class Mask(Tensor):
    """`HubFeature` for mask"""

    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        max_shape: Tuple[int, ...] = None,
        chunks=None,
        compressor="lz4",
    ):
        """Constructs a Mask HubFeature.

        shape: tuple of ints or None: (height, width, 1)
        dtype: dtype of mask array. Default: `uint8`

        Example:
        ```python
        mask_tensor = Mask(shape=(300, 300, 1))
        ```
        """
        super().__init__(
            shape,
            "bool_",
            max_shape=max_shape,
            chunks=chunks,
            compressor=compressor,
        )

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
