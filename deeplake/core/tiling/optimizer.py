import numpy as np
from typing import Tuple, Union, List, Optional


def get_tile_shape(
    sample_shape: Tuple[int, ...],
    sample_size: Optional[float] = None,
    chunk_size: int = 16 * 2**20,
    exclude_axes: Optional[Union[int, List[int]]] = None,
) -> Tuple[int, ...]:
    """
    Get tile shape for a given sample shape that will fit in chunk_size

    Args:
        sample_shape: Shape of the sample
        sample_size: Size of the compressed sample in bytes
        chunk_size: Expected size of a compressed tile in bytes
        exclude_axes: Dimensions to be excluded from tiling. (2 for RGB images)

    Returns:
        Tile shape

    Raises:
        ValueError: If the chunk_size is too small
    """
    ratio = sample_size / chunk_size  # type: ignore
    sample_shape = np.array(sample_shape, dtype=np.float32)  # type: ignore
    if isinstance(exclude_axes, int):
        exclude_axes = [exclude_axes]
    elif exclude_axes is None:
        exclude_axes = []
    elif not isinstance(exclude_axes, list):
        # should be a list for numpy advanced indexing
        exclude_axes = list(exclude_axes)
    sample_shape_masked = sample_shape.copy()  # type: ignore
    sample_shape_masked[exclude_axes] = 0
    while ratio > 1:
        idx = np.argmax(sample_shape_masked)
        val = sample_shape_masked[idx : idx + 1]  # type: ignore
        if val < 2:
            raise ValueError(f"Chunk size is too small: {chunk_size} bytes")
        val /= 2
        ratio /= 2
    sample_shape_masked[exclude_axes] = sample_shape[exclude_axes]  # type: ignore
    arr = np.ceil(sample_shape_masked)
    # convert arr to a tuple of python integers
    return tuple(int(x) for x in arr)
