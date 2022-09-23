from typing import Optional, Tuple, Union, List, Any
from hub.core.partial_sample import PartialSample
import numpy as np


def tiled(
    sample_shape: Tuple[int, ...],
    tile_shape: Optional[Tuple[int, ...]] = None,
    dtype: Union[str, np.dtype] = np.dtype("uint8"),
):
    """Allocates an empty sample of shape ``sample_shape``, broken into tiles of shape ``tile_shape`` (except for edge tiles).

    Example:

        >>> with ds:
        ...    ds.create_tensor("image", htype="image", sample_compression="png")
        ...    ds.image.append(hub.tiled(sample_shape=(1003, 1103, 3), tile_shape=(10, 10, 3)))
        ...    ds.image[0][-217:, :212, 1:] = np.random.randint(0, 256, (217, 212, 2), dtype=np.uint8)

    Args:
        sample_shape (Tuple[int, ...]): Full shape of the sample.
        tile_shape (Optional, Tuple[int, ...]): The sample will be will stored as tiles where each tile will have this shape (except edge tiles).
            If not specified, it will be computed such that each tile is close to half of the tensor's `max_chunk_size` (after compression).
        dtype (Union[str, np.dtype]): Dtype for the sample array. Default uint8.

    Returns:
        PartialSample: A PartialSample instance which can be appended to a Tensor.
    """
    return PartialSample(sample_shape=sample_shape, tile_shape=tile_shape, dtype=dtype)
