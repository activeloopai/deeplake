import collections.abc as abc
from typing import Tuple
import posixpath
import json

import numpy as np

import hub.store.storage_tensor as storage_tensor
from hub.store.store import get_fs_and_path, get_storage_map

from hub.exceptions import DynamicTensorNotFoundException
from hub.api.dataset_utils import slice_extract_info

StorageTensor = storage_tensor.StorageTensor

Shape = Tuple[int, ...]


class DynamicTensor:
    """Class for handling dynamic tensor

    This class adds dynamic nature to storage tensor.
    The shape of tensor depends only on the index of the first dim.
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
        """Constructor
        Parameters
        ----------
        url : str
            Path of folder where tensor should be stored, can be remote s3, gcs, ... path
        mode : str
            Mode in which tensor is opened (default is "r"), can be used to overwrite or append
        shape : Tuple[int | None]
            Shape of tensor, (must be specified) can contains Nones meaning the shape might change
        max_shape: Tuple[int | None]
            Maximum possible shape of the tensor (must be specified)
        dtype : str
            Numpy analog dtype for this tensor
        token : str | dict
            Access token for s3 or gcs resource. Could be anything referencing to credentials
            Could be filepath, str, dict
        memcache : int
            Number of bytes to use for caching this tensor
        chunks : Tuple[int] | True
            How to split the tensor into chunks (files) (default is True)
            If chunks=True then chunksize will automatically be detected
        fs : fsspec.AbstractFileSystem
            What filesystem use for read/write
            if not specified it will be generated from url and token
        fs_map : MutableMap
            Maps filesystem to MutableMap
            If not specified will be generated from fs.
            Note there is multiple ways so far to specify filesystem access
            url + token, fs and url in that fs, fs_map
        """
        fs, path = (fs, url) if fs else get_fs_and_path(url, token=token)
        if ("w" in mode or "a" in mode) and not fs.exists(path):
            fs.makedirs(path)
        fs_map = fs_map or get_storage_map(fs, path, memcache)
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
                # FIXME throw an error and explain whats wrong
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
        ranged_slice_count = len([i for i in slice_[1:] if isinstance(i, slice)])
        if real_shapes is not None:
            for r, i in enumerate(self._dynamic_dims):
                if i >= len(slice_):
                    real_shapes[r] = value.shape[i - len(slice_) + ranged_slice_count]
                else:
                    real_shapes[r] = max(
                        real_shapes[r], self._get_slice_upper_boundary(slice_[i])
                    )
        if self._dynamic_tensor:
            self.set_shape(slice_, value)
        slice_ += [slice(0, None, 1) for i in self.max_shape[len(slice_) :]]
        slice_ = self._get_slice(slice_, real_shapes)
        self._storage_tensor[slice_] = value
        if real_shapes is not None:
            self._dynamic_tensor[slice_[0]] = real_shapes

    def get_shape(self, slice_):
        """Gets the shape of the slice from tensor"""
        if self._dynamic_tensor is None:
            return self.shape
        final_shape = []
        shape_offset = 0
        for item in slice_[1:]:
            if isinstance(item, slice):
                final_shape += [item.stop - item.start]
            shape_offset += 1
        final_shape += list(self._dynamic_tensor[slice_[0]][shape_offset:])
        return tuple(final_shape)

    def set_shape(self, slice_, value):
        """Sets the shape of the slice of tensor"""
        if isinstance(slice_[0], slice):
            num, ofs = slice_extract_info(slice_[0], self.shape[0])
            slice_[0] = ofs if num == 1 else slice_[0]
        if isinstance(slice_[0], int):
            value_shape = list(value.shape) if isinstance(value, np.ndarray) else [1]
            new_shape = []
            shape_offset = 0
            for i in range(1, len(slice_)):
                new_shape.append(0)
                if isinstance(slice_[i], slice):
                    shape_offset += 1
            new_shape += value_shape[shape_offset:]
            self._dynamic_tensor[slice_[0]] = list(
                np.maximum(self._dynamic_tensor[slice_[0]], new_shape)
            )
        else:
            raise ValueError("Setting shape across multiple dimensions isn't supported")

    def _get_slice(self, slice_, real_shapes):
        # Makes slice_ which is uses relative indices (ex [:-5]) into precise slice_ (ex [10:40])
        slice_ = list(slice_)
        if real_shapes is not None:
            for r, i in enumerate(self._dynamic_dims):
                if isinstance(slice_[i], int) and slice_[i] < 0:
                    slice_[i] += real_shapes[r]
                elif isinstance(slice_[i], slice) and (
                    slice_[i].stop is None or slice_[i].stop < 0
                ):
                    slice_[i] = slice_stop_changed(
                        slice_[i], (slice_[i].stop or 0) + real_shapes[r]
                    )
        return tuple(slice_)

    @classmethod
    def _get_slice_upper_boundary(cls, slice_):
        if isinstance(slice_, slice):
            return slice_.stop
        else:
            assert isinstance(slice_, int)
            return slice_ + 1

    @property
    def chunksize(self):
        """
        Get chunk shape of the array
        """
        return self._storage_tensor.chunks

    def _get_chunking_dim(self):
        for i, d in enumerate(self.chunksize):
            if d != 1:
                return i, self.shape[i], self.chunksize[i]
        return 0, self.shape[0], self.chunksize[0]

    def chunk_slice_iterator(self):
        """
        Get an iterator over chunk coordinates
        """
        # FIXME assume chunking is done in one dimension
        nth, shpd, chnkd = self._get_chunking_dim()
        n_els = int(shpd / chnkd)
        for i in range(n_els):
            yield [1] * nth + [slice(i * chnkd, (i + 1) * chnkd)]

    def chunk_iterator(self):
        """
        Get an iterator over chunks
        """
        slices = self.chunk_slice_iterator()
        for slice_chunk in slices:
            yield self.__getitem__(*slice_chunk)

    def commit(self):
        self._storage_tensor.commit()
        if self._dynamic_tensor:
            self._dynamic_tensor.commit()


def get_dynamic_dims(shape):
    return [i for i, s in enumerate(shape) if s is None]


def slice_stop_changed(slice_, new_stop):
    return slice(slice_.start, new_stop, slice_.step)
