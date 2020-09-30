from typing import Tuple
from pathlib import posixpath

from hub.features import featurify, FeatureConnector, FlatTensor
from hub.store.storage_tensor import StorageTensor
import hub.collections.dataset.core as core
import json
import hub.features.serialize
import hub.features.deserialize
import hub.dynamic_tensor as dynamic_tensor

DynamicTensor = dynamic_tensor.DynamicTensor


class Dataset:
    def __init__(
        self,
        url: str = None,
        token=None,
        num_samples: int = None,
        mode: str = None,
        dtype=None,
    ):
        assert dtype is not None
        assert num_samples is not None
        assert url is not None
        assert mode is not None

        self.url = url
        self.token = token
        self.mode = mode

        fs, path = core._load_fs_and_path(self.url, creds=token)
        if mode in ["r", "w+"] and fs.exists(path + "/meta.json"):
            with fs.open(path + "/meta.json", "r") as f:
                meta = json.loads(f.read())
            self.num_samples = meta["num_samples"]
            self.dtype = hub.features.deserialize.deserialize(meta["dtype"])
            self._flat_tensors: Tuple[FlatTensor] = tuple(self.dtype._flatten())
            self._tensors = dict(self._open_storage_tensors())
        else:
            self.dtype: FeatureConnector = featurify(dtype)
            self.num_samples = num_samples
            meta = {
                "num_samples": num_samples,
                "dtype": hub.features.serialize.serialize(self.dtype),
            }
            fs.makedirs(path, exist_ok=True)
            with fs.open(path + "/meta.json", "w") as f:
                f.write(json.dumps(meta))
            self._flat_tensors: Tuple[FlatTensor] = tuple(self.dtype._flatten())
            self._tensors = dict(self._generate_storage_tensors())

    def _generate_storage_tensors(self):
        for t in self._flat_tensors:
            t: FlatTensor = t
            yield t.path, DynamicTensor(
                posixpath.join(self.url, t.path[1:]),
                shape=(self.num_samples,) + t.shape,
                max_shape=(self.num_samples,) + t.max_shape,
                dtype=t.dtype,
                chunks=t.chunks,
            )

    def _open_storage_tensors(self):
        for t in self._flat_tensors:
            t: FlatTensor = t
            yield t.path, DynamicTensor(
                posixpath.join(self.url, t.path[1:]),
                shape=(self.num_samples,) + t.shape,
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

    def commit(self):
        raise NotImplementedError()

    def __exit__(self, type, value, traceback):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


def open(
    url: str = None, token=None, num_samples: int = None, mode: str = None
) -> Dataset:
    raise NotImplementedError()
