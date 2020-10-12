from hub.features.features import Primitive
from typing import Tuple
import posixpath

# import fsspec

from hub.features import featurify, FeatureConnector, FlatTensor
# from hub.store.storage_tensor import StorageTensor
from hub.api.tensorview import TensorView
from hub.api.datasetview import DatasetView
from hub.api.dataset_utils import slice_extract_info, slice_split_tuple

# import hub.collections.dataset.core as core
import json
import hub.features.serialize
import hub.features.deserialize
import hub.dynamic_tensor as dynamic_tensor
import hub.utils as utils
from hub.exceptions import OverwriteIsNotSafeException
from hub.utils import MetaStorage

DynamicTensor = dynamic_tensor.DynamicTensor
MetaStorage = utils.MetaStorage


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
            path = posixpath.join(self._path, t.path[1:])
            self._fs.makedirs(path)
            yield t.path, DynamicTensor(
                path,
                mode=self.mode,
                shape=self.shape + t.shape,
                max_shape=self.shape + t.max_shape,
                dtype=t.dtype,
                chunks=t.chunks,
                fs=self._fs,
                fs_map=MetaStorage(t.path, utils.get_storage_map(self._fs, path), self._fs_map)
            )

    def _open_storage_tensors(self):
        for t in self._flat_tensors:
            t: FlatTensor = t
            path = posixpath.join(self._path, t.path[1:])
            yield t.path, DynamicTensor(
                path,
                mode=self.mode,
                shape=self.shape + t.shape,
                fs=self._fs,
                fs_map=MetaStorage(t.path, utils.get_storage_map(self._fs, path), self._fs_map)
            )

    def __getitem__(self, slice_):
        if isinstance(slice_, int):             # return Dataset with single sample
            # doesn't handle negative right now
            if slice_ >= self.shape[0]:
                raise IndexError('index out of bounds for dimension with length {}'.format(self.shape[0]))
            return DatasetView(dataset=self, num_samples=1, offset=slice_)

        elif isinstance(slice_, slice):         # return Dataset with multiple samples
            num, ofs = slice_extract_info(slice_, self.shape[0])
            return DatasetView(dataset=self, num_samples=num, offset=ofs)

        elif isinstance(slice_, str):
            subpath = slice_ if slice_.startswith("/") else "/" + slice_     # return TensorView object
            if subpath in self._tensors.keys():
                return TensorView(dataset=self, subpath=subpath, slice_=slice(0, self.shape[0]))
            else:
                d = {}
                post_subpath = subpath if subpath.endswith("/") else subpath + "/"
                for key in self._tensors.keys():
                    if key.startswith(post_subpath):
                        suffix_key = key[len(post_subpath):]
                    else:
                        continue
                    split_key = suffix_key.split("/")
                    cur = d
                    for i in range(len(split_key) - 1):
                        if split_key[i] in cur.keys():
                            cur = cur[split_key[i]]
                        else:
                            cur[split_key[i]] = {}
                            cur = cur[split_key[i]]
                    cur[split_key[-1]] = TensorView(dataset=self, subpath=key, slice_=slice(0, self.shape[0]))
                if len(d) == 0:
                    raise KeyError(f"Key {subpath} was not found in dataset")
                return d

        elif isinstance(slice_, tuple):        # return tensor view object
            subpath, slice_ = slice_split_tuple(slice_)
            if len(slice_) == 0:
                slice_ = (slice(0, self.shape[0]),)
            d = {}
            if subpath not in self._tensors.keys():
                post_subpath = subpath if subpath.endswith("/") else subpath + "/"
                for key in self._tensors.keys():
                    if key.startswith(post_subpath):
                        suffix_key = key[len(post_subpath):]
                    else:
                        continue
                    split_key = suffix_key.split("/")
                    cur = d
                    for i in range(len(split_key) - 1):
                        if split_key[i] in cur.keys():
                            cur = cur[split_key[i]]
                        else:
                            cur[split_key[i]] = {}
                            cur = cur[split_key[i]]
                    cur[split_key[-1]] = TensorView(dataset=self, subpath=key, slice_=slice_)
                if len(d) == 0:
                    raise KeyError(f"Key {subpath} was not found in dataset")

            if len(slice_) <= 1:
                if len(slice_) == 1 :
                    if isinstance(slice_[0], int) and slice_[0] >= self.shape[0]:
                        raise IndexError('index out of bounds for dimension with length {}'.format(self.shape[0]))
                    elif isinstance(slice_[0], slice):
                        # will check slice limits and raise error if required
                        num, ofs = slice_extract_info(slice_[0], self.shape[0])
                if subpath in self._tensors.keys():
                    return TensorView(dataset=self, subpath=subpath, slice_=slice_)
                else:
                    return d
            else:
                if subpath not in self._tensors.keys():
                    raise ValueError("You can't slice a dictionary of Tensors")
                elif isinstance(slice_[0], int) and slice_[0] >= self.shape[0]:
                    raise IndexError('index out of bounds for dimension with length {}'.format(self.shape[0]))
                elif isinstance(slice_[0], slice):
                    num, ofs = slice_extract_info(slice_[0], self.shape[0])
                    ls = list(slice_)
                    ls[0] = slice(ofs, ofs + num)
                    slice_ = tuple(ls)
                return TensorView(dataset=self, subpath=subpath, slice_=slice_)
        else:
            raise TypeError("type {} isn't supported in dataset slicing".format(type(slice_)))

    def __setitem__(self, slice_, value):
        if isinstance(slice_, int):             # Not supported
            raise TypeError("Can't assign to dataset indexed only with int")

        elif isinstance(slice_, slice):         # Not supported
            raise TypeError("Can't assign to dataset indexed only with slice")

        elif isinstance(slice_, str):
            slice_ = slice_ if slice_.startswith("/") else "/" + slice_     # return TensorView object
            self._tensors[slice_][:] = value

        elif isinstance(slice_, tuple):        # return tensor view object
            subpath, slice_ = slice_split_tuple(slice_)
            self._tensors[subpath][slice_] = value

        else:
            raise TypeError("type {} isn't supported in dataset slicing".format(type(slice_)))

    def __exit__(self, type, value, traceback):
        raise NotImplementedError()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.shape[0]

    def commit(self):
        for t in self._tensors.values():
            t.commit()


def open(
    url: str = None, token=None, num_samples: int = None, mode: str = None
) -> Dataset:
    raise NotImplementedError()
