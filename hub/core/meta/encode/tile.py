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

        self.entries[idx] = {
            "sample_shape": shape,
            "tile_shape": tile_shape,
            "chunks": [],  # TODO: maybe we can get away with storing this information strictly in the chunk_id_encoder?
        }

    def register_chunks_for_sample(self, idx: int, chunks: List[str]):
        if idx not in self.entries:
            raise ValueError(
                "Index not found. Entry must be registered before being populated."
            )

        self.entries[idx]["chunks"].extend(chunks)

    def chunk_for_sample(self, sample_idx: int, index: Tuple[int, ...]):
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

    @property
    def nbytes(self):
        # TODO: BEFORE MERGING IMPLEMENT THIS PROPERLY
        return 100

    def __getstate__(self) -> Dict[str, Any]:
        return {"entries": self.entries}
