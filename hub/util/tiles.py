from hub.core.meta.tensor_meta import TensorMeta
from typing import Tuple
import numpy as np


def _num_bytes_without_compression(shape: Tuple[int], dtype: np.dtype) -> int:
    return dtype.itemsize * np.prod(shape)


def _approximate_num_bytes_after_compressing(num_bytes: int, compressor: str) -> int:
    # TODO: come up with better approximations for compression amount, also add support for all of our compressors
    if compressor == "png":
        return num_bytes // 2
    elif compressor == "mp4":
        return num_bytes // 100
    else:
        raise NotImplementedError


def approximate_num_bytes(shape, tensor_meta: TensorMeta) -> int:
    """Calculate the number of bytes required to store raw data with the given shape. If no compression is used, this will be an exact
    number of bytes. If compressed, it will be approximated assuming the data is natural."""

    num_bytes = _num_bytes_without_compression(shape, np.dtype(tensor_meta.dtype))

    # check sample compression first. we don't support compressing both sample + chunk-wise at the same time, but in case we
    # do support this in the future, try both.
    if tensor_meta.sample_compression is not None:
        num_bytes = _approximate_num_bytes_after_compressing(
            num_bytes, tensor_meta.sample_compression
        )

    # TODO: UNCOMMENT AFTER CHUNK-WISE COMPRESSION IS MERGED!
    # if tensor_meta.chunk_compression is not None:
    #     num_bytes = _approximate_num_bytes_after_compressing(
    #         num_bytes, tensor_meta.chunk_compression
    #     )

    return num_bytes


def num_tiles_for_sample(tile_shape: Tuple[int], sample_shape: Tuple[int]):
    # TODO: docstring

    raise NotImplementedError  # TODO