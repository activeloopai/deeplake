from typing import Tuple
import posixpath
import collections.abc as abc

from hub.features.features import Primitive, Tensor, FeatureDict, FeatureConnector, featurify, FlatTensor

from hub.api.tensorview import TensorView
from hub.api.datasetview import DatasetView
from hub.api.dataset_utils import slice_extract_info, slice_split
from hub.utils import compute_lcm
import json
import hub.features.serialize
import hub.features.deserialize

from hub.store.dynamic_tensor import DynamicTensor
from hub.store.store import get_fs_and_path, get_storage_map
from hub.exceptions import OverwriteIsNotSafeException
from hub.store.metastore import MetaStorage
try:
    import torch
except ImportError:
    pass
try:
    import tensorflow as tf
except ImportError:
    pass


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

        fs, path = (fs, url) if fs else get_fs_and_path(self.url, token=token)
        if ("w" in mode or "a" in mode) and not fs.exists(path):
            fs.makedirs(path)
        fs_map = fs_map or get_storage_map(fs, path, 2 ** 20)
        self._fs = fs
        self._path = path
        self._fs_map = fs_map
        exist_ = bool(fs_map.get(".hub.dataset"))
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
                "version": 1,
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
                fs_map=MetaStorage(
                    t.path, get_storage_map(self._fs, path), self._fs_map
                ),
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
                fs_map=MetaStorage(
                    t.path, get_storage_map(self._fs, path), self._fs_map
                ),
            )

    def __getitem__(self, slice_):
        """Gets a slice or slices from dataset"""
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        if not subpath:
            if len(slice_list) > 1:
                raise ValueError(
                    "Can't slice a dataset with multiple slices without subpath"
                )
            num, ofs = slice_extract_info(slice_list[0], self.shape[0])
            return DatasetView(dataset=self, num_samples=num, offset=ofs)
        elif not slice_list:
            if subpath in self._tensors.keys():
                return TensorView(
                    dataset=self, subpath=subpath, slice_=slice(0, self.shape[0])
                )
            return self._get_dictionary(subpath)
        else:
            num, ofs = slice_extract_info(slice_list[0], self.shape[0])
            if subpath in self._tensors.keys():
                return TensorView(dataset=self, subpath=subpath, slice_=slice_list)
            if len(slice_list) > 1:
                raise ValueError("You can't slice a dictionary of Tensors")
            return self._get_dictionary(subpath, slice_list[0])

    def __setitem__(self, slice_, value):
        """"Sets a slice or slices with a value"""
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_, all_slices=False)
        if not subpath:
            raise ValueError("Can't assign to dataset sliced without subpath")
        elif not slice_list:
            self._tensors[subpath][:] = value  # Add path check
        else:
            self._tensors[subpath][slice_list] = value

    def to_pytorch(self, Transform=None):
        return TorchDataset(self, Transform)

    def to_tensorflow(self):
        def tf_gen():
            for index in range(self.shape[0]):
                d = {}
                for key in self._tensors.keys():
                    split_key = key.split("/")
                    cur = d
                    for i in range(1, len(split_key) - 1):
                        if split_key[i] in cur.keys():
                            cur = cur[split_key[i]]
                        else:
                            cur[split_key[i]] = {}
                            cur = cur[split_key[i]]
                    cur[split_key[-1]] = self._tensors[key][index]
                yield(d)

        def dict_to_tf(my_dtype):
            d = {}
            for k, v in my_dtype.dict_.items():
                d[k] = dtype_to_tf(v)
            return d

        def tensor_to_tf(my_dtype):
            return dtype_to_tf(my_dtype.dtype)

        def dtype_to_tf(my_dtype):
            if isinstance(my_dtype, FeatureDict):
                return dict_to_tf(my_dtype)
            elif isinstance(my_dtype, Tensor):
                return tensor_to_tf(my_dtype)
            elif isinstance(my_dtype, Primitive):
                return str(my_dtype._dtype)
            else:
                print(my_dtype, type(my_dtype), type(Tensor), isinstance(my_dtype, Tensor))

        output_types = dtype_to_tf(self.dtype)
        print(output_types)
        return tf.data.Dataset.from_generator(
            tf_gen,
            output_types=output_types,
        )

    def _get_dictionary(self, subpath, slice_=None):
        """"Gets dictionary from dataset given incomplete subpath"""
        tensor_dict = {}
        subpath = subpath if subpath.endswith("/") else subpath + "/"
        for key in self._tensors.keys():
            if key.startswith(subpath):
                suffix_key = key[len(subpath) :]
                split_key = suffix_key.split("/")
                cur = tensor_dict
                for i in range(len(split_key) - 1):
                    if split_key[i] not in cur.keys():
                        cur[split_key[i]] = {}
                    cur = cur[split_key[i]]
                slice_ = slice_ if slice_ else slice(0, self.shape[0])
                cur[split_key[-1]] = TensorView(
                    dataset=self, subpath=key, slice_=slice_
                )
        if len(tensor_dict) == 0:
            raise KeyError(f"Key {subpath} was not found in dataset")
        return tensor_dict

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.shape[0]

    def commit(self):
        for t in self._tensors.values():
            t.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.commit()

    @property
    def chunksize(self):
        # FIXME assumes chunking is done on the first sample
        chunks = [t.chunksize[0] for t in self._tensors.values()]
        return compute_lcm(chunks)


def open(
    url: str = None, token=None, num_samples: int = None, mode: str = None
) -> Dataset:
    raise NotImplementedError()


class TorchDataset:
    def __init__(self, ds, transform=None):
        self._ds = ds
        self._transform = transform

    def _do_transform(self, data):
        return self._transform(data) if self._transform else data

    def __len__(self):
        return self._ds.shape[0]

    def __getitem__(self, index):
        d = {}
        for key in self._ds._tensors.keys():
            split_key = key.split("/")
            cur = d
            for i in range(1, len(split_key) - 1):
                if split_key[i] in cur.keys():
                    cur = cur[split_key[i]]
                else:
                    cur[split_key[i]] = {}
                    cur = cur[split_key[i]]
            cur[split_key[-1]] = torch.tensor(self._ds._tensors[key][index])
        return d

    def __iter__(self):
        for index in range(self.shape[0]):
            d = {}
            for key in self._ds._tensors.keys():
                split_key = key.split("/")
                cur = d
                for i in range(1, len(split_key) - 1):
                    if split_key[i] in cur.keys():
                        cur = cur[split_key[i]]
                    else:
                        cur[split_key[i]] = {}
                        cur = cur[split_key[i]]
                cur[split_key[-1]] = torch.tensor(self._tensors[key][index])
            yield(d)

