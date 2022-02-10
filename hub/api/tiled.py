from typing import Optional, Tuple, Union, List, Any
from hub.core.partial_sample import PartialSample
import numpy as np


def tiled(
    sample_shape: Tuple[int, ...],
    tile_shape: Tuple[int, ...],
    dtype: Union[str, np.dtype] = np.uint8,
):
    return PartialSample(sample_shape=sample_shape, tile_shape=tile_shape, dtype=dtype)
