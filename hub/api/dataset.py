from typing import Tuple
import posixpath

# import fsspec

from hub.features import featurify, FeatureConnector, FlatTensor
# from hub.store.storage_tensor import StorageTensor
from hub.api.tensorview import TensorView
# import hub.collections.dataset.core as core
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
        offset=0,
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
        self.offset = offset

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

    def __getitem__(self, slice_):
        if isinstance(slice_, int):             # return Dataset with single sample
            # doesn't handle negative right now
            if slice_ >= self.shape[0]:
                raise IndexError('index out of bounds for dimension with length {}'.format(self.shape[0]))
            return Dataset(url=self.url, token=self.token, shape=(1,), mode=self.mode , dtype=self.dtype, offset=self.offset + slice_)

        elif isinstance(slice_, slice):         # return Dataset with multiple samples
            num, ofs = slice_extract_info(slice_, self.shape[0])
            return Dataset(url=self.url, token=self.token, shape=(num,), mode=self.mode , dtype=self.dtype, offset=self.offset + ofs)

        elif isinstance(slice_, str):
            subpath = slice_ if slice_.startswith("/") else "/" + slice_     # return TensorView object
            # add slice for original Dataset range
            offset_slice = slice(self.offset, self.offset + self.shape[0])
            return TensorView(dataset=self, subpath=subpath, slice_=offset_slice)

        elif isinstance(slice_, tuple):        # return tensor view object
            subpath, slice_ = slice_split_tuple(slice_)

            if len(slice_) == 0:
                slice_ = slice(self.offset, self.offset + self.shape[0])
            elif isinstance(slice_[0], int):
                if slice_[0] >= self.shape[0]:
                    raise IndexError('index out of bounds for dimension with length {}'.format(self.shape[0]))
                ls = list(slice_)
                ls[0] += self.offset
                slice_ = tuple(ls)
            elif isinstance(slice_[0], slice):
                num, ofs = slice_extract_info(slice_[0], self.shape[0])
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
            self._tensors[slice_][slice(self.offset, self.offset + self.shape[0])] = value

        elif isinstance(slice_, tuple):        # return tensor view object
            subpath, slice_ = slice_split_tuple(slice_)
            if len(slice_) == 0:
                slice_ = slice(self.offset, self.offset + self.shape[0])
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


def slice_extract_info(slice_, num):
    assert isinstance(slice_, slice)

    if slice_.step is not None and slice_.step < 0:       # negative step not supported
        raise ValueError("Negative step not supported in dataset slicing")

    offset = 0

    if slice_.start is not None:
        if slice_.start < 0:                 # make indices positive if possible
            slice_.start += num
            if slice_.start < 0:
                raise IndexError('index out of bounds for dimension with length {}'.format(num))

        if slice_.start >= num:
            raise IndexError('index out of bounds for dimension with length {}'.format(num))
        offset = slice_.start

    if slice_.stop is not None:
        if slice_.stop < 0:                   # make indices positive if possible
            slice_.stop += num
            if slice_.stop < 0:
                raise IndexError('index out of bounds for dimension with length {}'.format(num))

        if slice_.stop >= num:
            raise IndexError('index out of bounds for dimension with length {}'.format(num))

    if slice_.start is not None and slice_.stop is not None:
        if slice_.stop < slice_.start:    # return empty
            num = 0
        else:
            num = slice_.stop - slice_.start
    elif slice_.start is None and slice_.stop is not None:
        num = slice_.stop
    elif slice_.start is not None and slice_.stop is None:
        num = num - slice_.start
    else:
        num = num

    return num, offset
