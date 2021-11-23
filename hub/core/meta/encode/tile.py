import hub
import numpy as np
from hub.core.storage.cachable import Cachable

from typing import Any, Dict, Tuple


class TileEncoder(Cachable):
    def __init__(self, entries=None):
        self.entries = entries or {}
        self.version = hub.__version__

    def register_sample(
        self, idx: int, shape: Tuple[int, ...], tile_shape: Tuple[int, ...]
    ):
        """Registers a new tiled sample into the encoder.

        Args:
            idx: The global sample index.
            shape: The shape of the sample.
            tile_shape: The shape of the tiles of the sample.
        """
        # TODO: htype-based tile ordering?
        self.entries[str(idx)] = {
            "sample_shape": shape,
            "tile_shape": tile_shape,  # TODO: maybe this should be dynamic?
        }

    def __getitem__(self, global_sample_index: int):
        return self.entries[str(global_sample_index)]

    def __contains__(self, global_sample_index: int):
        """Returns whether the index is present in the tile encoder. Useful for checking if a given sample is tiled."""
        return str(global_sample_index) in self.entries

    def get_tile_shape(self, global_sample_index: int):
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

        if len(tile_shape) != len(sample_shape):
            raise ValueError(
                "Tile shape and sample shape must have the same number of dimensions."
            )

        layout = [
            np.ceil(sample_shape_dim / tile_shape_dim)
            for tile_shape_dim, sample_shape_dim in zip(tile_shape, sample_shape)
        ]
        return tuple(int(x) for x in layout)

    @property
    def nbytes(self):
        # TODO: optimize this
        return len(self.tobytes())

    def __getstate__(self) -> Dict[str, Any]:
        return {"entries": self.entries, "version": self.version}

    def __setstate__(self, state: Dict[str, Any]):
        self.entries = state["entries"]
        self.version = state["version"]
