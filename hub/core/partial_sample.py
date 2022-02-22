from typing import Tuple, Optional, Union
import numpy as np


class PartialSample:
    """Represents a sample that is initialized by just shape and the data is updated later."""

    def __init__(
        self,
        sample_shape: Tuple[int, ...],
        tile_shape: Optional[Tuple[int, ...]] = None,
        dtype: Union[str, np.dtype] = np.dtype("uint8"),
    ):
        self.sample_shape = sample_shape
        self.tile_shape = tile_shape
        self.dtype = dtype

    @property
    def shape(self):
        return self.sample_shape

    def astype(self, dtype: Union[str, np.dtype]):
        return self.__class__(self.sample_shape, self.tile_shape, dtype)
