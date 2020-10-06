from hub.features.features import Primitive
from typing import Tuple
import posixpath

import fsspec

from hub.features import featurify, FeatureConnector, FlatTensor
from hub.store.storage_tensor import StorageTensor
import hub.collections.dataset.core as core
import json
import hub.features.serialize
import hub.features.deserialize
import hub.dynamic_tensor as dynamic_tensor
import hub.utils as utils
from hub.exceptions import OverwriteIsNotSafeException

DynamicTensor = dynamic_tensor.DynamicTensor


class Dataset:
    def __init__(
        self,
        url: str = None,
        mode: str = None,
        token=None,
        shape=None,
        dtype=None,
        fs=None,
        fs_map=None,
    ):
        assert dtype is not None
        assert shape is not None
        assert len(tuple(shape)) == 1
        assert url is not None
        assert mode is not None

        self.url = url
        self.token = token
        self.mode = mode

        fs, path = (fs, url) if fs else utils.get_fs_and_path(self.url, token=token)
        if ("w" in mode or "a" in mode) and not fs.exists(path):
            fs.makedirs(path)
        fs_map = fs_map or utils.get_storage_map(fs, path, 2 ** 20)
        self._fs = fs
        self._path = path
        self._fs_map = fs_map
        exist_ = fs_map.get(".hub.dataset")
        if not exist_ and len(fs_map) > 0 and "w" in mode:
            raise OverwriteIsNotSafeException()
        if len(fs_map) > 0 and exist_ and "w" in mode:
            fs.rm(path, recursive=True)
            fs.makedirs(path)
        exist = False if "w" in mode else bool(fs_map.get(".hub.dataset"))
        if exist:
            meta = json.loads(str(fs_map[".hub.dataset"]))
            self.shape = meta["shape"]
            self.dtype = hub.features.deserialize.deserialize(meta["dtype"])
            self._flat_tensors: Tuple[FlatTensor] = tuple(self.dtype._flatten())
            self._tensors = dict(self._open_storage_tensors())
        else:
            self.dtype: FeatureConnector = featurify(dtype)
            self.shape = shape
            meta = {
                "shape": shape,
                "dtype": hub.features.serialize.serialize(self.dtype),
            }
            fs_map[".hub.dataset"] = bytes(json.dumps(meta), "utf-8")
            self._flat_tensors: Tuple[FlatTensor] = tuple(self.dtype._flatten())
            self._tensors = dict(self._generate_storage_tensors())

    def _generate_storage_tensors(self):
        for t in self._flat_tensors:
            t: FlatTensor = t
            yield t.path, DynamicTensor(
                posixpath.join(self._path, t.path[1:]),
                mode=self.mode,
                shape=self.shape + t.shape,
                max_shape=self.shape + t.max_shape,
                dtype=t.dtype,
                chunks=t.chunks,
                fs=self._fs,
            )

    def _open_storage_tensors(self):
        for t in self._flat_tensors:
            t: FlatTensor = t
            yield t.path, DynamicTensor(
                posixpath.join(self._path, t.path[1:]),
                mode=self.mode,
                shape=self.shape + t.shape,
                fs=self._fs,
            )

    def _slice_split(self, slice_):
        path = slice_[0]
        assert isinstance(path, str)
        slice_ = slice_[1:]
        path = path if path.startswith("/") else "/" + path
        return path, slice_

    def __getitem__(self, slice_):
        path, slice_ = self._slice_split(slice_)
        return self._tensors[path][slice_]

    def __setitem__(self, slice_, value):
        path, slice_ = self._slice_split(slice_)
        self._tensors[path][slice_] = value

    def __exit__(self, type, value, traceback):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def commit(self):
        for t in self._tensors.values():
            t.commit()


def open(
    url: str = None, token=None, num_samples: int = None, mode: str = None
) -> Dataset:
    raise NotImplementedError()
