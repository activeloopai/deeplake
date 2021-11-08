from typing import List, Optional, Tuple, Union
import numpy as np
from hub.core.tiling.optimizer import get_tile_shape
from hub.core.tiling.serialize_tile import break_into_tiles, serialize_tiles

compression_ratios = {None: 1, "jpeg": 0.5, "png": 0.5, "webp": 0.5}


class SampleTiles:
    def __init__(self, arr: np.ndarray, compression: str, chunk_size):
        self.arr = arr
        self.compression = compression
        self.sample_shape = arr.shape

        self.tile_shape = get_tile_shape(
            arr.shape, arr.nbytes * compression_ratios[compression], chunk_size, 2
        )
        tiles = break_into_tiles(arr, self.tile_shape)
        self.tiles = serialize_tiles(tiles, lambda x: memoryview(x.tobytes()))
        self.registered = False
        # product of self.tiles.shape
        self.num_tiles = self.tiles.size
        self.tiles_yielded = 0
        self.tiles_enumerator = np.ndenumerate(self.tiles)

    def is_first_write(self):
        return self.tiles_yielded == 1

    def is_last_write(self):
        return self.tiles_yielded == self.num_tiles

    def yield_tile(self):
        self.tiles_yielded += 1
        return next(self.tiles_enumerator)[1]
