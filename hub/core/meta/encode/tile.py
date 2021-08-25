import numpy as np

from hub.constants import ENCODING_DTYPE
from hub.core.index.index import Index
from hub.util.tiles import ceildiv
from hub.core.storage.cachable import Cachable

from typing import Any, Dict, List, Tuple



def get_tile_layout_shape(tile_shape: Tuple[int], sample_shape: Tuple[int]) -> Tuple[int]:
    # TODO: docstring

    assert len(tile_shape) == len(sample_shape), 'need same dimensionality'  # TODO: exception (sanity check)

    layout = []
    for tile_shape_dim, sample_shape_dim in zip(tile_shape, sample_shape):
        layout.append(ceildiv(sample_shape_dim, tile_shape_dim))

    return tuple(layout)

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

    def translate_index_relative_to_tiles(self, element_index: Tuple[int], tile_shape: Tuple[int], sample_shape: Tuple[int]) -> int:
        """Takes in an N-dimensional element-index and returns a 1d index to be used for a flat tile-ordered list of chunks"""

        # TODO: remove this method?

        raise NotImplementedError


    # def prune_chunks(self, tile_ids: List[ENCODING_DTYPE], global_sample_index: int, subslice_index: Index) -> List[ENCODING_DTYPE]:
    #     """This method handles the main tile logic, given a subslice_index it replaces the chunk IDs that 
    #     are not needed to be downloaded with `None`, returning a new (pruned) list.

    #     Args:
    #         tile_ids (List[ENCODING_DTYPE]): All tile chunk IDs that correspond with this sample in tile-order.
    #         global_sample_index (int): Primary index for the tensor.
    #             Example: `tensor[0, 10:50, 50:100]` -- the first slice component would be the `global_sample_index`.
    #         subslice_index (Index): Subslice index of a tensor. 
    #             Example: `tensor[0, 10:50, 50:100]` -- the last 2 slice components would be the `subslice_index`.

    #     Raises:
    #         IndexError: Must first call `register_sample`.

    #     Returns:
    #         List[ENCODING_DTYPE]: New list same length & ordering, except the chunks that aren't requird for `subslice_index` are
    #             `None` values.
    #     """

    #     if global_sample_index not in self.entries:
    #         raise IndexError(f"Global sample index {global_sample_index} does not exist in tile encoder.")

    #     # TODO: return a new list of the same length with only needed chunks (otherwise they're None)
    #     # TODO: add a sanity check for len(tile_ids) (make sure it's rootable)

    #     tile_meta = self.entries[global_sample_index]
    #     tile_shape = tile_meta["tile_shape"]
    #     sample_shape = tile_meta["sample_shape"]
    #     num_tiles = len(tile_ids)

    #     pruned_tile_ids = []
    #     for tile_index_1d, tile_id in enumerate(tile_ids):
    #         tile_origin_element_index = self.first_element_index_of_tile(tile_shape, num_tiles, tile_index_1d)
    #         tile_inside_subslice = self.is_element_index_inside_subslice(tile_origin_element_index, subslice_index)

    #         if tile_inside_subslice:
    #             pruned_tile_ids.append(tile_id)
    #         else:
    #             pruned_tile_ids.append(None)
    #             
    #     return tile_ids

    
    # def first_element_index_of_tile(self, tile_shape: Tuple[int], num_tiles: int, tile_index_1d: int) -> Tuple[int]:
    #     # TODO: docstring

    #     raise NotImplementedError


    def order_tiles(self, global_sample_index: int, chunk_ids: List[ENCODING_DTYPE]) -> np.ndarray:
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

        tile_meta = self.entries[global_sample_index]
        tile_shape = tile_meta["tile_shape"]
        sample_shape = tile_meta["sample_shape"]

        tile_layout_shape = get_tile_layout_shape(tile_shape, sample_shape)

        ordered_tiles = np.array(chunk_ids, dtype=ENCODING_DTYPE)
        ordered_tiles = np.reshape(ordered_tiles, tile_layout_shape)

        return ordered_tiles


    def get_tile_shape_mask(self, sample_index: int, ordered_tile_ids: np.ndarray) -> np.ndarray:
        # TODO: docstring

        if sample_index not in self.entries:
            return np.array([])

        tile_meta = self.entries[sample_index]
        tile_shape = tile_meta["tile_shape"]

        tile_shape_mask = np.empty(ordered_tile_ids.shape, dtype=object)

        # right now tile shape is the same for all tiles, but we might want to add dynamic tile shapes
        # also makes lookup easier later
        for tile_index, _ in np.ndenumerate(ordered_tile_ids):
            tile_shape_mask[tile_index] = tile_shape

        return tile_shape_mask


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
