import numpy as np

import hub
from hub.constants import ENCODING_DTYPE
from hub.util.tiles import ceildiv
from hub.core.storage.cachable import Cachable

from typing import Any, Dict, List, Tuple


# TODO: do we want to make this a BaseEncoder subclass?
class TileEncoder(Cachable):
    def __init__(self, entries=None):
        self.entries = entries or {}
        self.version = hub.__version__

    def register_sample(
        self, idx: int, shape: Tuple[int, ...], tile_shape: Tuple[int, ...]
    ):
        # TODO: docstring

        # TODO: htype-based tile ordering?
        self.entries[str(idx)] = {
            "sample_shape": shape,
            "tile_shape": tile_shape,  # TODO: maybe this should be dynamic?
        }

    def __getitem__(self, global_sample_index: int):
        return self.entries[str(global_sample_index)]

    def __contains__(self, global_sample_index: int):
        return str(global_sample_index) in self.entries

    def get_tile_shape(self, global_sample_index: int):
        # TODO: maybe this should be dynamic?
        return tuple(self[global_sample_index]["tile_shape"])

    def get_sample_shape(self, global_sample_index: int):
        return tuple(self[global_sample_index]["sample_shape"])

    def get_tile_layout_shape(self, global_sample_index: int) -> Tuple[int, ...]:
        """If you were to lay the tiles out in a grid, the tile layout shape would be the shape
        of the grid.

        Example:
            Sample shape:               (1000, 500)
            Tile shape:                 (10, 10)
            Output tile layout shape:   (100, 50)
        """

        tile_meta = self[global_sample_index]
        tile_shape = tile_meta["tile_shape"]
        sample_shape = tile_meta["sample_shape"]

        assert len(tile_shape) == len(
            sample_shape
        ), "need same dimensionality"  # TODO: exception (sanity check)

        layout = []
        for tile_shape_dim, sample_shape_dim in zip(tile_shape, sample_shape):
            layout.append(ceildiv(sample_shape_dim, tile_shape_dim))

        return tuple(layout)

    def order_tiles(
        self, global_sample_index: int, chunk_ids: List[ENCODING_DTYPE]
    ) -> np.ndarray:
        """Given a flat list of `chunk_ids` for the sample at `global_sample_index`,
        return a new numpy array that has the tiles laid out how they will be
        spacially if they were on a single tensor.

        Example:
            Given 16 tiles that represent a 160x160 element sample in c-order:
                - each tile represents a 10x10 collection of elements.
                - should return:
                    [
                        [ch0, ch1, ch2, ch3],
                        [ch4, ch5, ch6, ch7],
                        [ch8, ch9, ch10, ch11],
                        [ch12, ch13, ch14, ch15],
                    ]
        """

        if len(chunk_ids) == 1:
            return np.array(chunk_ids)

        tile_layout_shape = self.get_tile_layout_shape(global_sample_index)

        ordered_tiles = np.array(chunk_ids, dtype=ENCODING_DTYPE)
        ordered_tiles = np.reshape(ordered_tiles, tile_layout_shape)

        return ordered_tiles

    def get_tile_shape_mask(
        self, global_sample_index: int, ordered_tile_ids: np.ndarray
    ) -> np.ndarray:
        # TODO: docstring

        if global_sample_index not in self:
            return np.array([])

        tile_shape = self.get_tile_shape(global_sample_index)
        tile_shape_mask = np.empty(ordered_tile_ids.shape, dtype=object)

        # right now tile shape is the same for all tiles, but we might want to add dynamic tile shapes
        # also makes lookup easier later
        for tile_index, _ in np.ndenumerate(ordered_tile_ids):
            tile_shape_mask[tile_index] = tile_shape

        return tile_shape_mask

    def chunk_index_for_tile(self, sample_index: int, tile_index: Tuple[int, ...]):
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
        # TODO: optimize this
        return len(self.tobytes())

    def __getstate__(self) -> Dict[str, Any]:
        return {"entries": self.entries, "version": self.version}

    def __setstate__(self, state: Dict[str, Any]):
        self.entries = state["entries"]
        self.version = state["version"]
