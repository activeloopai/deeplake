from hub.core.index.index import Index
from hub.util.tiles import ceildiv
from hub.core.storage.cachable import Cachable

from typing import Any, Dict, List, Tuple


# TODO: do we want to make this a BaseEncoder subclass?
class TileEncoder(Cachable):
    def __init__(self, entries=None):
        self.entries = entries or {}

    def register_sample(
        self, idx: int, shape: Tuple[int, ...], tile_shape: Tuple[int, ...]
    ):
        # TODO: docstring

        # TODO: htype-based tile ordering?
        self.entries[idx] = {
            "sample_shape": shape,
            "tile_shape": tile_shape,  # TODO: maybe this should be dynamic?
        }


    def prune_chunks(self, chunks: List, sample_index: int, subslice_index: Index):
        # TODO: docstring

        if sample_index not in self.entries:
            raise IndexError(f"Sample index {sample_index} does not exist in tile encoder.")

        # TODO: return a new list of chunks that exclude the 

        return chunks


    def chunk_index_for_tile(self, sample_index: int, tile_index: Tuple[int]):
        tile_meta = self.entries[sample_index]
        sample_shape = tile_meta["sample_shape"]
        tile_shape = tile_meta["tile_shape"]
        ndims = len(sample_shape)

        # Generalized row-major ordering
        chunk_idx = 0
        factor = 1
        for ax in range(ndims):
            chunk_idx += (tile_index[ax] // tile_shape[ax]) * factor
            factor *= ceildiv(tile_shape[ax], sample_shape[ax])
        
        return chunk_idx

    @property
    def nbytes(self):
        # TODO: BEFORE MERGING IMPLEMENT THIS PROPERLY
        return 100

    def __getstate__(self) -> Dict[str, Any]:
        return {"entries": self.entries}
