from typing import Tuple

import dask
import numpy as np

from hub.collections._store_version import CURRENT_STORE_VERSION
from hub.collections._chunk_utils import _tensor_chunksize, _logify_chunksize


def _dask_shape_backward(shape: Tuple[int]):
    if len(shape) == 0:
        return shape
    else:
        return (-1,) + (shape[1:]) if np.isnan(shape[0]) else shape


class Tensor:
    def __init__(self, meta: dict, daskarray, delayed_objs: tuple = None):
        if not meta.get("preprocessed"):
            meta = Tensor._preprocess_meta(meta, daskarray)
        self._meta = meta
        self._array = daskarray
        self._delayed_objs = delayed_objs
        self._shape = _dask_shape_backward(daskarray.shape)
        self._dtype = meta["dtype"]
        self._dtag = meta.get("dtag")
        self._dcompress = meta.get("dcompress")
        self._dcompress_algo = meta.get("dcompress_algo")
        self._dcompress_lvl = meta.get("dcompress_lvl")
        self._chunksize = meta.get("chunksize")

    @staticmethod
    def _preprocess_meta(meta, daskarray):
        meta = dict(meta)
        meta["preprocessed"] = True
        meta["dtype"] = meta.get("dtype") or daskarray.dtype
        if meta.get("chunksize") is None:
            if str(meta["dtype"]) == "object":
                meta["chunksize"] = 1
            else:
                meta["chunksize"] = _tensor_chunksize(daskarray)
        else:
            meta["chunksize"] = _logify_chunksize(meta["chunksize"])
        meta["shape"] = daskarray.shape
        meta["STORE_VERSION"] = CURRENT_STORE_VERSION
        if "dcompress" in meta:
            dcompress_comp = str(meta["dcompress"]).split(sep=":")
            assert len(dcompress_comp) in [
                1,
                2,
            ], "Invalid dcompress format, should be {algo:compress_lvl} or {algo}"
            meta["dcompress_algo"] = dcompress_comp[0]
            meta["dcompress_lvl"] = (
                dcompress_comp[1] if len(dcompress_comp) == 2 else None
            )
        else:
            meta["dcompress"] = None
            meta["dcompress_algo"] = None
            meta["dcompress_lvl"] = None
        return meta

    @property
    def meta(self):
        """
        Returns
        -------
        dict
            metadata dict of tensor
        """
        return dict(self._meta)

    @property
    def shape(self):
        """ 
        Returns
        -------
        tuple
            Tensor shape
        """
        return tuple(self._shape)

    @property
    def ndim(self):
        """
        Returns
        -------
        int
            Number of dimensions (len(shape))
        """
        return len(self._shape)

    @property
    def count(self):
        """
        Returns
        -------
        int
            Number of elements on axis 0 if that number is known, -1 otherwise
        """
        return self._shape[0]

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            Number of elements on axis 0 if that number is known, raises Exception otherwise
        """
        if self._shape[0] == -1:
            raise Exception(
                "__len__ can only return >=0 numbers, use .count property if the len of tensor is unknown (in which case -1 is returned)"
            )
        return self._shape[0]

    @property
    def dtype(self):
        """
        Returns
        -------
        str
            Tensor data type, equivalent of numpy dtype
        """
        return self._dtype

    @property
    def dtag(self):
        """
        Returns
        -------
        str
            Information about data stored in tensor (iamge, mask, label, ...)
        """
        return self._dtag

    @property
    def dcompress(self):
        return self._dcompress

    @property
    def chunksize(self):
        return self._chunksize

    def __getitem__(self, slices) -> "Tensor":
        """ Slices tensor
        Parameters
        ----------
        slices
            tuple of slices or ints

        Returns
        -------
        Tensor
            sliced tensor
        """
        if self.count == -1:
            raise Exception("Cannot slice an array with unknown len")
        arr = self._array[slices]
        if isinstance(arr, dask.delayed.__class__):
            assert False, "This branch should have never been reached, check me please"
            return arr
        else:
            return Tensor(self._meta, arr)

    def __iter__(self):
        # TODO this function should return tensors, not dask arrays
        for i in range(len(self)):
            yield self._array[i]

    def compute(self):
        """ Does lazy computation and converts data to numpy array
        Returns
        -------
        np.ndarray
            numpy array of tensor's data
        """
        return self._array.compute()
