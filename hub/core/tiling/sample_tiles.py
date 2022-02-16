from typing import Optional, Tuple, Union
import numpy as np

from hub.core.compression import compress_array
from hub.core.tiling.optimizer import get_tile_shape
from hub.core.tiling.serialize import break_into_tiles, serialize_tiles, get_tile_shapes
from hub.util.compression import get_compression_ratio
from hub.compression import BYTE_COMPRESSIONS
from hub.constants import MB


class SampleTiles:
    """Stores the tiles corresponding to a sample."""

    def __init__(
        self,
        arr: Optional[np.ndarray] = None,
        compression: Optional[str] = None,
        chunk_size: int = 16 * MB,
        store_uncompressed_tiles: bool = False,
        htype: Optional[str] = None,
        tile_shape: Optional[Tuple[int, ...]] = None,
        sample_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[Union[np.dtype, str]] = None,
    ):
        self.arr = arr
        self.compression = compression
        ratio = get_compression_ratio(compression)
        if arr is None:
            self.sample_shape = sample_shape
            nbytes = np.prod(sample_shape) * dtype  # type: ignore
        else:
            self.sample_shape = arr.shape
            nbytes = arr.nbytes
        if tile_shape is None:
            # Exclude channels axis from tiling for image, video and audio
            exclude_axis = (
                None
                if htype == "generic"
                and (not compression or compression in BYTE_COMPRESSIONS)
                else -1
            )

            self.tile_shape = get_tile_shape(
                self.sample_shape, nbytes * ratio, chunk_size, exclude_axis
            )
        else:
            self.tile_shape = tile_shape

        self.registered = False
        self.tiles_yielded = 0
        if arr is not None:

            tiles = break_into_tiles(arr, self.tile_shape)

            self.tiles = serialize_tiles(
                tiles, lambda x: memoryview(compress_array(x, self.compression))
            )
            tile_shapes = np.vectorize(lambda x: x.shape, otypes=[object])(tiles)

            self.shapes_enumerator = np.ndenumerate(tile_shapes)
            self.layout_shape = self.tiles.shape
            self.num_tiles = self.tiles.size
            self.tiles_enumerator = np.ndenumerate(self.tiles)
            self.uncompressed_tiles_enumerator = (
                np.ndenumerate(tiles) if store_uncompressed_tiles else None
            )

        else:
            self.tiles = None
            tile_shapes = get_tile_shapes(self.sample_shape, self.tile_shape)
            self.shapes_enumerator = np.ndenumerate(tile_shapes)
            self.layout_shape = tile_shapes.shape
            self.num_tiles = tile_shapes.size
            self.uncompressed_tiles_enumerator = None

    @property
    def is_first_write(self) -> bool:
        return self.tiles_yielded == 1

    @property
    def is_last_write(self) -> bool:
        return self.tiles_yielded == self.num_tiles

    def yield_tile(self):
        self.tiles_yielded += 1
        if self.tiles is None:
            tile = b""
        else:
            tile = next(self.tiles_enumerator)[1]
        return tile, next(self.shapes_enumerator)[1]

    def yield_uncompressed_tile(self):
        if self.uncompressed_tiles_enumerator is not None:
            return next(self.uncompressed_tiles_enumerator)[1]
