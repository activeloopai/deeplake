from typing import Optional, Tuple, Union, List, Any
from hub.core.partial_sample import PartialSample
import numpy as np


def tiled(
    sample_shape: Tuple[int, ...],
    tile_shape: Optional[Tuple[int, ...]] = None,
    dtype: Union[str, np.dtype] = np.dtype("uint8"),
):
    return PartialSample(sample_shape=sample_shape, tile_shape=tile_shape, dtype=dtype)
