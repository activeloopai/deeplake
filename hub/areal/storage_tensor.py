import typing
import math

import zarr
import numpy as np

import hub.areal.tensor
import hub.areal.store


class StorageTensor(hub.areal.tensor.Tensor):
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
        shape: typing.Tuple[int, ...] = None,
        dtype="float32",
        creds=None,
        memcache: float = None,
    ):
        if shape is not None:
            self._zarr = zarr.zeros(
                shape,
                dtype=dtype,
                chunks=self._determine_chunksizes(shape, dtype),
                store=hub.areal.store.get_storage_map(url, creds, memcache),
                overwrite=True,
            )
        else:
            self._zarr = zarr.open_array(
                hub.areal.store.get_storage_map(url, creds, memcache)
            )
        self._shape = self._zarr.shape
        self._chunks = self._zarr.chunks
        self._dtype = self._zarr.dtype
        self._memcache = memcache

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
