from hub.exceptions import StorageTensorNotFoundException
import typing
import math
import json
import zarr
import numpy as np
from hub.store.store import get_fs_and_path, get_storage_map

from hub.store.tensor import Tensor


class StorageTensor(Tensor):
    @classmethod
    def _determine_chunksizes(cls, shape, dtype, block_size=2 ** 24):
        """
        Autochunking of tensors
        Chunk is determined by 16MB blocks keeping left dimensions inside a chunk
        Dimensions from left are kept until 16MB block is filled

        Parameters
        ----------
        shape: tuple
            the shape of the whole array
        dtype: type
            the type of the element (int, float)
        block_size: int (optional)
            how big the chunk size should be. Default to 16MB
        """

        sz = np.dtype(dtype).itemsize
        elem_count_in_chunk = block_size / sz

        # Get left most part which will be left static inside the chunk
        a = list(shape)
        a.reverse()
        left_part = shape
        prod = 1
        for i, dim in enumerate(a):
            prod *= dim
            if elem_count_in_chunk < prod:
                left_part = shape[-i:]
                break

        # If the tensor is smaller then the chunk size return
        if len(left_part) == len(shape):
            return tuple(left_part)

        # Get the middle chunk size of dimension
        els = math.ceil(elem_count_in_chunk / cls._tuple_product(left_part))

        # Contruct the chunksize shape
        chunksize = [els] + list(left_part)
        if len(chunksize) < len(shape):
            chunksize = [[1] * (len(shape) - len(chunksize))] + chunksize
        return tuple(chunksize)

    @classmethod
    def _tuple_product(cls, tuple: typing.Tuple[int, ...]):
        res = 1
        for t in tuple:
            res *= t
        return res

    def __init__(
        self,
        url: str,
        mode: str = "r",
        shape=None,
        dtype=np.float64,
        token=None,
        memcache=2 ** 26,
        chunks=True,
        fs=None,
        fs_map=None,
    ):
        fs, path = (fs, url) if fs else get_fs_and_path(url, token=token)
        if ("w" in mode or "a" in mode) and not fs.exists(path):
            fs.makedirs(path)
        fs_map = fs_map or get_storage_map(fs, path, memcache)

        exist_ = bool(fs_map.get(".hub.storage_tensor"))
        # if not exist_ and len(fs_map) > 0 and "w" in mode:
        #     raise OverwriteIsNotSafeException()
        exist = False if "w" in mode else exist_
        if "r" in mode and not exist:
            raise StorageTensorNotFoundException()

        if "r" in mode or "a" in mode and exist:
            self._zarr = zarr.open_array(fs_map)
        else:
            self._zarr = zarr.zeros(
                shape,
                dtype=dtype,
                chunks=self._determine_chunksizes(shape, dtype)
                if chunks is True
                else chunks,
                store=fs_map,
                overwrite=("w" in mode),
            )
            fs_map[".hub.storage_tensor"] = bytes(json.dumps(dict()), "utf-8")
        self._shape = self._zarr.shape
        self._chunks = self._zarr.chunks
        self._dtype = self._zarr.dtype
        self._memcache = memcache
        self._fs = fs
        self._fs_map = fs_map
        self._path = path

    def __getitem__(self, slice_):
        return self._zarr[slice_]

    def __setitem__(self, slice_, value):
        self._zarr[slice_] = value

    @property
    def shape(self):
        return self._shape

    @property
    def chunks(self):
        return self._chunks

    @property
    def dtype(self):
        return self._dtype

    def commit(self):
        self._fs_map.commit()
