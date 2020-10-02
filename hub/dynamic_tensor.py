import collections.abc as abc
from typing import Tuple
import posixpath
import json

import numpy as np

import hub.collections.dataset.core as core
import hub.store.storage_tensor as storage_tensor
import hub.utils as utils

from hub.exceptions import DynamicTensorNotFoundException, OverwriteIsNotSafeException

StorageTensor = storage_tensor.StorageTensor

Shape = Tuple[int, ...]


class DynamicTensor:
    """Class for handling dynamicness of the tensor

    This class adds dynamic nature to storage tensor.
    The shape of tensor depends on the index of the first dim.
    """

    # TODO Make first dim is extensible as well
    def __init__(
        self,
        url: str,
        mode: str = "r",
        shape: Shape = None,
        max_shape: Shape = None,
        dtype="float64",
        token=None,
        memcache=None,
        chunks=True,
        fs=None,
        fs_map=None,
    ):
        fs, path = (fs, url) if fs else utils.get_fs_and_path(url, token=token)
        if ("w" in mode or "a" in mode) and not fs.exists(path):
            fs.makedirs(path)
        fs_map = fs_map or utils.get_storage_map(fs, path, memcache)
        exist_ = fs_map.get(".hub.dynamic_tensor")
        # if not exist_ and len(fs_map) > 0 and "w" in mode:
        #     raise OverwriteIsNotSafeException()
        exist = False if "w" in mode else exist_ is not None
        if exist:
            meta = json.loads(str(fs_map.get(".hub.dynamic_tensor")))
            shape = meta["shape"]
        else:
            meta = {"shape": shape}
            fs_map[".hub.dynamic_tensor"] = bytes(json.dumps(meta), "utf-8")
        self._dynamic_dims = get_dynamic_dims(shape)
        if "r" in mode and not exist:
            raise DynamicTensorNotFoundException()

        if ("r" in mode or "a" in mode) and exist:
            self._storage_tensor = StorageTensor(path, mode=mode, fs=fs, fs_map=fs_map)
        else:
            self._storage_tensor = StorageTensor(
                path,
                mode=mode,
                shape=max_shape,
                dtype=dtype,
                chunks=chunks,
                fs=fs,
                fs_map=fs_map,
            )

        if ("r" in mode or "a" in mode) and exist:
            self._dynamic_tensor = (
                StorageTensor(
                    posixpath.join(path, "dynamic"),
                    mode=mode,
                    memcache=2 ** 25,
                    fs=fs,
                    fs_map=fs_map,
                )
                if self._dynamic_dims
                else None
            )
        else:
            self._dynamic_tensor = (
                StorageTensor(
                    posixpath.join(path, "dynamic"),
                    mode=mode,
                    shape=(max_shape[0], len(self._dynamic_dims)),
                    dtype=np.int32,
                    fs=fs,
                    memcache=2 ** 25,
                )
                if self._dynamic_dims
                else None
            )
        self.shape = shape
        self.max_shape = self._storage_tensor.shape
        self.dtype = self._storage_tensor.dtype
        assert len(self.shape) == len(self.max_shape)
        for item in self.max_shape:
            assert item is not None
        for item in zip(self.shape, self.max_shape):
            if item[0] is not None:
                assert item[0] == item[1]

    def __getitem__(self, slice_):
        """Gets a slice or slices from tensor"""
        if not isinstance(slice_, abc.Iterable):
            slice_ = [slice_]
        slice_ = list(slice_)
        # real_shapes is dynamic shapes based on first dim index, only dynamic dims are stored, static ones are ommitted
        if self._dynamic_tensor:
            real_shapes = self._dynamic_tensor[slice_[0]]
        else:
            real_shapes = None
        # Extend slice_ to dim count
        slice_ += [slice(0, None, 1) for i in self.max_shape[len(slice_) :]]
        slice_ = self._get_slice(slice_, real_shapes)
        return self._storage_tensor[slice_]

    def __setitem__(self, slice_, value):
        """Sets a slice or slices with a value"""
        if not isinstance(slice_, abc.Iterable):
            slice_ = [slice_]
        slice_ = list(slice_)
        real_shapes = self._dynamic_tensor[slice_[0]] if self._dynamic_tensor else None
        if real_shapes is not None:
            for r, i in enumerate(self._dynamic_dims):
                if i >= len(slice_):
                    real_shapes[r] = value.shape[i - len(slice_)]
        slice_ += [slice(0, None, 1) for i in self.max_shape[len(slice_) :]]
        slice_ = self._get_slice(slice_, real_shapes)
        self._storage_tensor[slice_] = value
        if real_shapes is not None:
            self._dynamic_tensor[slice_[0]] = real_shapes

    def _get_slice(self, slice_, real_shapes):
        # Makes slice_ which is uses relative indices (ex [:-5]) into precise slice_ (ex [10:40])
        slice_ = list(slice_)
        if real_shapes is not None:
            for r, i in enumerate(self._dynamic_dims):
                if isinstance(slice_[i], int) and slice_[i] < 0:
                    slice_[i] += real_shapes[i]
                elif isinstance(slice_[i], slice) and (
                    slice_[i].stop is None or slice_[i].stop < 0
                ):
                    slice_[i] = slice_stop_changed(
                        slice_[i], (slice_[i].stop or 0) + real_shapes[r]
                    )
        return tuple(slice_)


def get_dynamic_dims(shape):
    return [i for i, s in enumerate(shape) if s is None]


def slice_stop_changed(slice_, new_stop):
    return slice(slice_.start, new_stop, slice_.step)
