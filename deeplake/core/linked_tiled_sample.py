from typing import Optional
import numpy as np
from deeplake.core.tensor_link import read_linked_sample
from deeplake.util.path import get_path_type


class LinkedTiledSample:
    """Represents a sample that is initialized using external links. See :meth:`deeplake.link_tiled`."""

    def __init__(
        self,
        path_array: np.ndarray,
        creds_key: Optional[str] = None,
    ):
        self.path_array = path_array
        self._path = None
        # check that all paths in the array are of the same type
        path_types = {get_path_type(path) for path in path_array.flat}
        if len(path_types) > 1:
            raise ValueError("Path array contains paths in different locations.")
        self.creds_key = creds_key
        self._tile_shape = None
        self._shape = None

    @property
    def path(self):
        if self._path is None:
            self._path = next(iter(self.path_array.flat))
        return self._path

    def dtype(self) -> str:
        return np.array("").dtype.name

    def set_check_tile_shape(self, link_creds, verify):
        tile_shape = None
        for path in self.path_array.flat:
            # check that path is string like
            if not isinstance(path, str):
                raise ValueError("Path array contains non-string paths.")
            sample_obj = read_linked_sample(path, self.creds_key, link_creds, verify)
            if tile_shape is None:
                tile_shape = sample_obj.shape
                if not verify:
                    break
            elif tile_shape != sample_obj.shape:
                raise ValueError("Path array contains paths with different shapes.")
        self._tile_shape = tile_shape
        if len(self.path_array.shape) > len(self._tile_shape):
            raise ValueError(
                "Path array can not contain more dimensions than the individual tiles"
            )

    def set_sample_shape(self):
        assert self._tile_shape is not None
        if len(self.path_array.shape) < len(self._tile_shape):
            arr_shape = self.path_array.shape + (1,) * (
                len(self._tile_shape) - len(self.path_array.shape)
            )
            self.path_array = self.path_array.reshape(arr_shape)

        self._shape = tuple(
            arr_dim * tile_dim
            for arr_dim, tile_dim in zip(self.path_array.shape, self._tile_shape)
        )

    @property
    def shape(self):
        return self._shape

    @property
    def sample_shape(self):
        return self._shape

    @property
    def tile_shape(self):
        return self._tile_shape
