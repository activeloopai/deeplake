from typing import Tuple
from pathlib import posixpath

from hub.features import featurify, FeatureConnector, FlatTensor
# from hub.store.storage_tensor import StorageTensor
from hub.api.tensorview import TensorView
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
        offset=0,
        _tensors=None
    ):
        assert dtype is not None
        assert num_samples is not None
        assert url is not None or _tensors is not None
        assert mode is not None

        self.url = url
        self.token = token
        self.mode = mode
        self.offset = offset
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

    def __getitem__(self, slice_):
        if isinstance(slice_, int):             # return Dataset with single sample
            # doesn't handle negative right now
            if slice_ >= self.num_samples:
                raise IndexError('index out of bounds for dimension with length {}'.format(self.num_samples))
            return Dataset(url=self.url, token=self.token, num_samples=1, mode=self.mode , dtype=self.dtype, offset=self.offset + slice_)

        elif isinstance(slice_, slice):         # return Dataset with multiple samples
            num, ofs = slice_extract_info(slice_, self.num_samples)
            return Dataset(url=self.url, token=self.token, num_samples=num, mode=self.mode , dtype=self.dtype, offset=self.offset + ofs)

        elif isinstance(slice_, str):
            subpath = slice_ if slice_.startswith("/") else "/" + slice_     # return TensorView object
            # add slice for original Dataset range
            offset_slice = slice(self.offset, self.offset + self.num_samples)
            return TensorView(dataset=self, subpath=subpath, slice_=offset_slice)

        elif isinstance(slice_, tuple):        # return tensor view object
            subpath, slice_ = slice_split_tuple(slice_)

            if len(slice_) == 0:
                slice_ = slice(self.offset, self.offset + self.num_samples)
            elif isinstance(slice_[0], int):
                if slice_[0] >= self.num_samples:
                    raise IndexError('index out of bounds for dimension with length {}'.format(self.num_samples))
                ls = list(slice_)
                ls[0] += self.offset
                slice_ = tuple(ls)
            elif isinstance(slice_[0], slice):
                num, ofs = slice_extract_info(slice_[0], self.num_samples)
                ls = list(slice_)
                ls[0] = slice(self.offset + ofs, self.offset + ofs + num)
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
            self._tensors[slice_][slice(self.offset, self.offset + self.num_samples)] = value

        elif isinstance(slice_, tuple):        # return tensor view object
            subpath, slice_ = slice_split_tuple(slice_)
            if len(slice_) == 0:
                slice_ = slice(self.offset, self.offset + self.num_samples)
            elif isinstance(slice_[0], int):
                offset_slice = slice_[0] + self.offset
                ls = list(slice_)
                ls[0] = offset_slice
                slice_ = tuple(ls)
            elif isinstance(slice_[0], slice):
                offset_slice = slice(slice_[0].start + self.offset, slice_[0].stop + self.offset)
                ls = list(slice_)
                ls[0] = offset_slice
                slice_ = tuple(ls)
            self._tensors[subpath][slice_] = value

        else:
            raise TypeError("type {} isn't supported in dataset slicing".format(type(slice_)))

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


def slice_split_tuple(slice_):
    path = ""
    list_slice = []
    for sl in slice_:
        if isinstance(sl, str):
            path += sl if sl.startswith("/") else "/" + sl
        elif isinstance(sl, int) or isinstance(sl, slice):
            list_slice.append(sl)
        else:
            raise TypeError("type {} isn't supported in dataset slicing".format(type(sl)))
    tuple_slice = tuple(list_slice)
    if path == "":
        raise ValueError("No path found in slice!")
    return path, tuple_slice


def slice_extract_info(slice_, num_samples):
    assert isinstance(slice_, slice)

    if slice_.step is not None and slice_.step < 0:       # negative step not supported
        raise ValueError("Negative step not supported in dataset slicing")

    offset = 0

    if slice_.start is not None:
        if slice_.start < 0:                 # make indices positive if possible
            slice_.start += num_samples
            if slice_.start < 0:
                raise IndexError('index out of bounds for dimension with length {}'.format(num_samples))

        if slice_.start >= num_samples:
            raise IndexError('index out of bounds for dimension with length {}'.format(num_samples))
        offset = slice_.start

    if slice_.stop is not None:
        if slice_.stop < 0:                   # make indices positive if possible
            slice_.stop += num_samples
            if slice_.stop < 0:
                raise IndexError('index out of bounds for dimension with length {}'.format(num_samples))

        if slice_.stop >= num_samples:
            raise IndexError('index out of bounds for dimension with length {}'.format(num_samples))

    if slice_.start is not None and slice_.stop is not None:
        if slice_.stop < slice_.start:    # return empty
            num = 0
        else:
            num = slice_.stop - slice_.start
    elif slice_.start is None and slice_.stop is not None:
        num = slice_.stop
    elif slice_.start is not None and slice_.stop is None:
        num = num_samples - slice_.start
    else:
        num = num_samples

    return num, offset
