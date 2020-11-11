from typing import Tuple
# import numpy as np

from hub.features.features import Tensor
# from hub.features.polygon import Polygon


class Mask(Tensor):
    """`FeatureConnector` for mask

    | Usage:
    ----------
    >>> mask_tensor = Mask(shape=(300, 300, 1))
    """
    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        dtype=None,
        max_shape: Tuple[int, ...] = None,
        chunks=True
    ):
        """Constructs a Mask FeatureConnector.

        Parameters
        ----------
        shape: tuple of ints or None
            Shape in format (height, width, 1)
        dtype: str
            Dtype of mask array. Default: `uint8`
        max_shape : Tuple[int]
            Maximum shape of tensor shape if tensor is dynamic
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks
        
        """
        if not dtype:
            dtype = 'uint8'
        super(Mask, self).__init__(shape, dtype, max_shape=max_shape, chunks=chunks)

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
