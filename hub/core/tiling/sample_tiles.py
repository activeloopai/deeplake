from typing import Optional
import numpy as np

from hub.core.compression import compress_array
from hub.core.tiling.optimizer import get_tile_shape
from hub.core.tiling.serialize import break_into_tiles, serialize_tiles
from hub.util.compression import get_compression_ratio
from hub.compression import BYTE_COMPRESSIONS


class SampleTiles:
    """Stores the tiles corresponding to a sample."""

    def __init__(
        self,
        arr: np.ndarray,
        compression: Optional[str],
        chunk_size: int,
        store_uncompressed_tiles: bool = False,
        htype: Optional[str] = None,
    ):
        self.arr = arr
        self.compression = compression
        self.sample_shape = arr.shape
        ratio = get_compression_ratio(compression)

        # Exclude channels axis from tiling for image, video and audio
        exclude_axis = (
            None
            if htype == "generic"
            and (not compression or compression in BYTE_COMPRESSIONS)
            else -1
        )

        self.tile_shape = get_tile_shape(
            arr.shape, arr.nbytes * ratio, chunk_size, exclude_axis
        )
        tiles = break_into_tiles(arr, self.tile_shape)

        self.tiles = serialize_tiles(
            tiles, lambda x: compress_array(x, self.compression)
        )
        tile_shapes = np.vectorize(lambda x: x.shape, otypes=[object])(tiles)

        self.shapes_enumerator = np.ndenumerate(tile_shapes)
        self.layout_shape = self.tiles.shape
        self.registered = False
        self.num_tiles = self.tiles.size
        self.tiles_yielded = 0
        self.tiles_enumerator = np.ndenumerate(self.tiles)
        self.uncompressed_tiles_enumerator = (
            np.ndenumerate(tiles) if store_uncompressed_tiles else None
        )

    @property
    def is_first_write(self) -> bool:
        return self.tiles_yielded == 1

    @property
    def is_last_write(self) -> bool:
        return self.tiles_yielded == self.num_tiles

    def yield_tile(self):
        self.tiles_yielded += 1
        return next(self.tiles_enumerator)[1], next(self.shapes_enumerator)[1]

    def yield_uncompressed_tile(self):
        if self.uncompressed_tiles_enumerator is not None:
            return next(self.uncompressed_tiles_enumerator)[1]
