from hub.exceptions import (
    StorageTensorNotFoundException,
    OverwriteIsNotSafeException,
)
import posixpath
import typing
import math
import json

import zarr
import numpy as np

import hub.store.tensor
import hub.store.store
import hub.utils as utils


class StorageTensor(hub.store.tensor.Tensor):
    @classmethod
    def _determine_chunksizes(cls, shape, dtype):
        sz = np.dtype(dtype).itemsize
        elem_count_in_chunk = (2 ** 24) / sz
        ratio = (cls._tuple_product(shape) / elem_count_in_chunk) ** (1 / len(shape))
        return tuple([int(math.ceil(s / ratio)) for s in shape])

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
        fs, path = (fs, url) if fs else utils.get_fs_and_path(url, token=token)
        if ("w" in mode or "a" in mode) and not fs.exists(path):
            fs.makedirs(path)
        fs_map = fs_map or utils.get_storage_map(fs, path, memcache)
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
