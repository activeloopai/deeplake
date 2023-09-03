from typing import Optional, Tuple, Union
import numpy as np

from deeplake.core.compression import compress_array
from deeplake.core.tiling.optimizer import get_tile_shape
from deeplake.core.tiling.serialize import (
    break_into_tiles,
    serialize_tiles,
    get_tile_shapes,
)
from deeplake.util.compression import get_compression_ratio
from deeplake.compression import BYTE_COMPRESSIONS
from deeplake.constants import MB


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
        self.tiles_yielded = 0
        self.dtype = dtype
        if arr is not None:
            self._init_from_array(
                arr,
                compression,
                chunk_size,
                store_uncompressed_tiles,
                htype,
                tile_shape,
            )
        else:
            self._init_from_sample_shape(
                sample_shape,  # type: ignore
                compression,
                chunk_size,
                store_uncompressed_tiles,
                htype,
                tile_shape,
                dtype,
            )

    def _init_from_array(
        self,
        arr: np.ndarray,
        compression: Optional[str] = None,
        chunk_size: int = 16 * MB,
        store_uncompressed_tiles: bool = False,
        htype: Optional[str] = None,
        tile_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.arr = arr
        self.sample_shape = arr.shape
        tile_shape = tile_shape or self._get_tile_shape(
            arr.dtype, htype, chunk_size, compression  # type: ignore
        )
        self.tile_shape = tile_shape
        tiles = break_into_tiles(arr, tile_shape)
        self.tiles = serialize_tiles(tiles, lambda x: compress_array(x, compression))
        tile_shapes = np.vectorize(lambda x: x.shape, otypes=[object])(tiles)

        self.shapes_enumerator = np.ndenumerate(tile_shapes)
        self.layout_shape = self.tiles.shape
        self.num_tiles = self.tiles.size
        self.tiles_enumerator = np.ndenumerate(self.tiles)
        self.uncompressed_tiles_enumerator = (
            np.ndenumerate(tiles) if store_uncompressed_tiles else None
        )

    def _init_from_sample_shape(
        self,
        sample_shape: Tuple[int, ...],
        compression: Optional[str] = None,
        chunk_size: int = 16 * MB,
        store_uncompressed_tiles: bool = False,
        htype: Optional[str] = None,
        tile_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[Union[np.dtype, str]] = None,
    ):
        self.arr = None  # type: ignore
        self.sample_shape = sample_shape
        self.tiles = None  # type: ignore
        tile_shape = tile_shape or self._get_tile_shape(
            dtype, htype, chunk_size, compression  # type:ignore
        )
        self.tile_shape = tile_shape
        tile_shapes = get_tile_shapes(self.sample_shape, tile_shape)
        self.shapes_enumerator = np.ndenumerate(tile_shapes)
        self.layout_shape = tile_shapes.shape
        self.num_tiles = tile_shapes.size
        self.uncompressed_tiles_enumerator = None

    def _get_tile_shape(
        self, dtype: Union[np.dtype, str], htype: str, chunk_size: int, compression: str
    ):
        # Exclude channels axis from tiling for image, video and audio
        exclude_axis = (
            None
            if htype == "generic"
            and (not compression or compression in BYTE_COMPRESSIONS)
            else -1
        )
        return get_tile_shape(
            self.sample_shape,
            np.prod(np.array(self.sample_shape, dtype=np.uint64))  # type: ignore
            * np.dtype(dtype).itemsize
            * get_compression_ratio(compression),
            chunk_size,
            exclude_axis,
        )

    @property
    def is_first_write(self) -> bool:
        return self.tiles_yielded == 1

    @property
    def is_last_write(self) -> bool:
        return self.tiles_yielded == self.num_tiles

    @property
    def shape(self):
        return self.sample_shape

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
