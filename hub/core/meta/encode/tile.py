from hub.core.storage.cachable import Cachable
from hub.core.chunk import Chunk
from hub.constants import DEFAULT_MAX_CHUNK_SIZE

from typing import Tuple
import numpy as np
import math


def ceildiv(a: int, b: int) -> int:
    """Computes the ceiling of the division of two ints.
    Returns an int.
    """
    return math.ceil(float(a) / float(b))


class TileEncoder(Cachable):
    def __init__(self, entries=None):
        self.entries = entries or {}

    def register_sample(idx: int, shape: Tuple[int], dtype: np.dtype):
        tile_target_bytes = DEFAULT_MAX_CHUNK_SIZE
        dtype_num_bytes = dtype.itemsize
        ndims = len(shape)

        tile_target_entries = tile_target_bytes // dtype_num_bytes
        tile_target_length = int(tile_target_entries ** (1.0 / ndims))

        # Tile length must be at least 16, so we don't round too close to zero
        tile_shape = max(16, (tile_target_length,) * ndims)

        self.entries[idx] = {
            "sample_shape": shape,
            "tile_shape": tile_shape,
            "chunks": [],
        }

        num_chunks_needed = np.prod(map(ceildiv, zip(shape, tile_shape)))
        return num_chunks_needed

    def register_chunks_for_sample(idx: int, chunks: List[str]):
        if idx not in self.entries:
            raise ValueError(
                "Index not found. Entry must be registered before being populated."
            )

        self.entries[idx]["chunks"].extend(chunks)

    def chunk_for_sample(sample_idx: int, index: Tuple[int]):
        if sample_idx not in self.entries:
            return None

        sample_shape = self.entries[sample_idx]["sample_shape"]
        tile_shape = self.entries[sample_idx]["tile_shape"]
        chunks = self.entries[sample_idx]["chunks"]
        ndims = len(sample_shape)

        # Generalized row-major ordering
        chunk_idx = 0
        factor = 1
        for ax in range(ndims):
            chunk_idx += (index[ax] // tile_shape[ax]) * factor
            factor *= ceildiv(tile_shape[ax], sample_shape[ax])

        return chunks[chunk_idx]
