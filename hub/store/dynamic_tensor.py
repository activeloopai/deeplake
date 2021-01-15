import collections.abc as abc
from shutil import Error
from hub.schema.features import Shape
import json
import math

import numpy as np
from numpy.lib.arraysetops import isin
import zarr
import numcodecs

from hub.store.nested_store import NestedStore
from hub.store.shape_detector import ShapeDetector
from hub.defaults import DEFAULT_COMPRESSOR

from hub.exceptions import (
    DynamicTensorNotFoundException,
    ValueShapeError,
    DynamicTensorShapeException,
)
from hub.schema.sequence import Sequence


class DynamicTensor:
    """Class for handling dynamic tensor

    This class adds dynamic nature to storage tensor.
    The shape of tensor depends only on the index of the first dim.
    """

    # TODO Make first dim is extensible as well
    def __init__(
        self,
        fs_map: str,
        mode: str = "r",
        shape=None,
        max_shape=None,
        dtype="float64",
        chunks=None,
        compressor=DEFAULT_COMPRESSOR,
    ):
        """Constructor
        Parameters
        ----------
        fs_map : MutableMap
            Maps filesystem to MutableMap
        mode : str
            Mode in which tensor is opened (default is "r"), can be used to overwrite or append
        shape : Tuple[int | None]
            Shape of tensor, (must be specified) can contains Nones meaning the shape might change
        max_shape: Tuple[int | None]
            Maximum possible shape of the tensor (must be specified)
        dtype : str
            Numpy analog dtype for this tensor
        chunks : Tuple[int] | True
            How to split the tensor into chunks (files) (default is True)
            If chunks=True then chunksize will automatically be detected

        """
        if not (shape is None):
            # otherwise shape detector fails
            shapeDt = ShapeDetector(
                shape, max_shape, chunks, dtype, compressor=compressor
            )
            shape = shapeDt.shape
            max_shape = shapeDt.max_shape
            chunks = shapeDt.chunks
        elif "r" not in mode:
            raise TypeError("shape cannot be none")

        self.fs_map = fs_map
        exist_ = fs_map.get(".hub.dynamic_tensor")

        # if not exist_ and len(fs_map) > 0 and "w" in mode:
        #     raise OverwriteIsNotSafeException()
        exist = False if "w" in mode else exist_ is not None
        if "r" in mode and not exist:
            raise DynamicTensorNotFoundException()

        synchronizer = None
        # synchronizer = zarr.ThreadSynchronizer()
        # synchronizer = zarr.ProcessSynchronizer("~/activeloop/sync/example.sync")
        # if tensor exists and mode is read or append

        if ("r" in mode or "a" in mode) and exist:
            meta = json.loads(fs_map.get(".hub.dynamic_tensor").decode("utf-8"))
            shape = meta["shape"]
            self._dynamic_dims = get_dynamic_dims(shape)
            self._storage_tensor = zarr.open_array(
                store=fs_map, mode=mode, synchronizer=synchronizer
            )
            self._dynamic_tensor = (
                zarr.open_array(
                    NestedStore(fs_map, "--dynamic--"),
                    mode=mode,
                    synchronizer=synchronizer,
                )
                if self._dynamic_dims
                else None
            )
        # else we need to create or overwrite the tensor
        else:
            self._dynamic_dims = get_dynamic_dims(shape)
            self._storage_tensor = zarr.zeros(
                max_shape,
                dtype=dtype,
                chunks=chunks,
                store=fs_map,
                overwrite=("w" in mode),
                object_codec=numcodecs.Pickle(protocol=3)
                if str(dtype) == "object"
                else None,
                compressor=compressor,
                synchronizer=synchronizer,
            )
            self._dynamic_tensor = (
                zarr.zeros(
                    shape=(max_shape[0], len(self._dynamic_dims)),
                    mode=mode,
                    dtype=np.int32,
                    store=NestedStore(fs_map, "--dynamic--"),
                    synchronizer=synchronizer,
                    compressor=None,
                )
                if self._dynamic_dims
                else None
            )

            fs_map[".hub.dynamic_tensor"] = bytes(json.dumps({"shape": shape}), "utf-8")

        self.shape = shape
        self.max_shape = self._storage_tensor.shape
        self.chunks = self._storage_tensor.chunks
        self.dtype = self._storage_tensor.dtype

        if len(self.shape) != len(self.max_shape):
            raise DynamicTensorShapeException("length")
        for item in self.max_shape:
            if item is None:
                raise DynamicTensorShapeException("none")
        for item in zip(self.shape, self.max_shape):
            if item[0] is not None:
                if item[0] != item[1]:
                    raise DynamicTensorShapeException("not_equal")
        self._enabled_dynamicness = True

    def __getitem__(self, slice_):
        """Gets a slice or slices from tensor"""
        if not isinstance(slice_, abc.Iterable):
            slice_ = [slice_]
        slice_ = list(slice_)
        # real_shapes is dynamic shapes based on first dim index, only dynamic dims are stored, static ones are ommitted
        if self._dynamic_tensor:
            if isinstance(slice_[0], int):
                real_shapes = self._dynamic_tensor[slice_[0]]
            elif (
                slice_[0].stop is not None
                and slice_[0].stop - (slice_[0].start or 0) == 1
            ):
                real_shapes = self._dynamic_tensor[slice_[0].start]
            else:
                raise ValueError(
                    "Getting item across multiitem slices is not supported for tensors with dynamic shapes, access them item by item"
                )
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
        if self._dynamic_tensor and self._enabled_dynamicness:
            self.set_shape(slice_, value)
        slice_ += [slice(0, None, 1) for i in self.max_shape[len(slice_) :]]

        if self._dynamic_tensor and isinstance(slice_[0], int):
            real_shapes = self._dynamic_tensor[slice_[0]]
        elif self._dynamic_tensor and isinstance(slice_[0], slice):
            max_shape = value[0].shape
            for item in value:
                max_shape = tuple(max(value) for value in zip(max_shape, item.shape))
            for i in range(len(value)):
                pad = [
                    (0, max_shape[dim] - value[i].shape[dim])
                    for dim in range(value[i].ndim)
                ]
                value[i] = np.pad(value[i], pad)
            real_shapes = np.array(
                [
                    max_shape[i]
                    for i in range(len(max_shape))
                    if i + 1 in self._dynamic_dims
                ]
            )
        else:
            real_shapes = None

        if not self._enabled_dynamicness:
            real_shapes = (
                list(value.shape)
                if hasattr(value, "shape")
                else real_shapes
                if real_shapes is not None
                else [1]
            )

        slice_ = self._get_slice(slice_, real_shapes)
        value = self.check_value_shape(value, slice_)
        self._storage_tensor[slice_] = value

    def check_value_shape(self, value, slice_):
        """Checks if value can be set to the slice"""
        if None not in self.shape and self.dtype != "O":
            if not all([isinstance(sh, int) for sh in slice_]):
                expected_value_shape = tuple(
                    [
                        len(range(*slice_shape.indices(self.shape[i])))
                        for i, slice_shape in enumerate(slice_)
                        if not isinstance(slice_shape, int)
                    ]
                )

                if isinstance(value, list):
                    value = np.array(value)
                if isinstance(value, np.ndarray):
                    value_shape = [dim for dim in value.shape if dim != 1]
                    expected_shape = [dim for dim in expected_value_shape if dim != 1]
                    if value_shape != expected_shape:
                        raise ValueShapeError(expected_value_shape, value.shape)
                    else:
                        value = value.reshape(expected_value_shape)
            else:
                expected_value_shape = (1,)
                if isinstance(value, list):
                    value = np.array(value)
                if (
                    isinstance(value, np.ndarray)
                    and value.shape != expected_value_shape
                ):
                    raise ValueShapeError(expected_value_shape, value.shape)
        return value

    def _resize_shape(self, tensor: zarr.Array, size: int) -> None:
        """append first dimension of single array"""
        shape = list(tensor.shape)
        shape[0] = size
        tensor.resize(*shape)

    def resize_shape(self, size: int) -> None:
        """append shape of storage and dynamic tensors"""
        self.shape = (size,) + self.shape[1:]
        self.max_shape = (size,) + self.max_shape[1:]
        self._resize_shape(self._storage_tensor, size)

        if self._dynamic_tensor:
            self._resize_shape(self._dynamic_tensor, size)

        self.fs_map[".hub.dynamic_tensor"] = bytes(
            json.dumps({"shape": self.shape}), "utf-8"
        )

    def get_shape_samples(self, samples):
        """Gets full shape of dynamic_tensor(s)"""
        if isinstance(samples, int):
            shape, shape_offset = [], 0
            for i in range(1, len(self.shape)):
                if self.shape[i] is not None:
                    current = self.shape[i]
                else:
                    current = self._dynamic_tensor[samples][shape_offset]
                    shape_offset += 1
                shape.append(current)
            return np.array(shape)
        elif isinstance(samples, slice):
            shapes = self._dynamic_tensor[samples]
            for i in range(1, len(self.shape)):
                if self.shape[i] is not None:
                    shapes = np.insert(shapes, i - 1, self.shape[i], axis=1)
            return shapes
        elif isinstance(samples, list):
            shapes = np.array([self._dynamic_tensor[index] for index in samples])
            for i in range(1, len(self.shape)):
                if self.shape[i] is not None:
                    shapes = np.insert(shapes, i - 1, self.shape[i], axis=1)
            return shapes

    def combine_shape(self, shape, slice_):
        """Combines given shape with slice to get final shape"""
        if len(slice_) > shape.shape[-1]:
            raise ValueError("Slice can't be longer than shape")
        if shape.ndim == 1:  # single shape accessed
            new_shape = np.ones((0))
            for i in range(shape.shape[-1]):
                if i < len(slice_) and isinstance(slice_[i], slice):
                    start = slice_[i].start if slice_[i].start is not None else 0
                    stop = slice_[i].stop if slice_[i].stop is not None else shape[i]
                    sl = stop - start if stop != 0 else 0
                    new_shape = np.append(new_shape, sl)
                elif i >= len(slice_):
                    new_shape = np.append(new_shape, shape[i])
        else:  # slice of shapes accessed
            new_shape = np.ones(
                (shape.shape[0], 0)
            )  # new shape with rows equal to number of shapes accessed
            for i in range(shape.shape[-1]):
                if i < len(slice_) and isinstance(slice_[i], slice):
                    start = slice_[i].start if slice_[i].start is not None else 0
                    stop = slice_[i].stop
                    if stop is None:
                        sh = shape[:, i : i + 1] - start
                        sh[sh < 0] = 0  # if negative in shape, replace with 0
                        new_shape = np.append(new_shape, sh, axis=1)
                    else:
                        sl = stop - start if stop != 0 else 0
                        new_shape = np.insert(
                            new_shape, new_shape.shape[1], sl, axis=1
                        )  # inserted as last column
                elif i >= len(slice_):
                    new_shape = np.append(new_shape, shape[:, i : i + 1], axis=1)

        return np.array(new_shape).astype("int")

    def get_shape(self, slice_):

        """Gets the shape of the slice from tensor"""
        if isinstance(slice_, (int, slice)):
            slice_ = [slice_]
        if self._dynamic_tensor is None:  # returns 1D np array
            return self.combine_shape(np.array(self.shape), slice_)
        elif isinstance(slice_[0], int):  # returns 1D np array
            sample_shape = self.get_shape_samples(slice_[0])
            return self.combine_shape(sample_shape, slice_[1:])
        elif isinstance(slice_[0], (slice, list)):
            sample_shapes = self.get_shape_samples(slice_[0])
            final_shapes = self.combine_shape(sample_shapes, slice_[1:])
            if len(final_shapes) == 1:
                return np.insert(final_shapes[0], 0, 1)  # returns 1D np array
            return final_shapes  # returns 2D np array

    def set_shape(self, slice_, value):
        """
        Set shape of the dynamic tensor given value
        """
        if not self._enabled_dynamicness:
            return

        new_shape = self.get_shape_from_value(slice_, value)
        self.set_dynamic_shape(slice_, new_shape)

    def set_dynamic_shape(self, slice_, shape):
        """
        Set shape from the shape directly
        """
        self._dynamic_tensor[slice_[0]] = shape

    def get_shape_from_value(self, slice_, value):
        """
        create shape for multiple elements
        """
        if isinstance(slice_[0], int):
            new_shapes = self.create_shape(slice_, value)
            new_shapes = np.maximum(self._dynamic_tensor[slice_[0]], new_shapes)
        else:
            start = slice_[0].start if slice_[0].start is not None else 0
            stop = (
                slice_[0].stop if slice_[0].stop is not None else start + value.shape[0]
            )
            dt = self._dynamic_tensor[slice_[0]]
            new_shapes = []
            for i in range(start, stop):
                new_shape = self.create_shape([i] + slice_[1:], value[i - start])
                new_shape = np.maximum(dt[i - start], new_shape)
                new_shapes.append(new_shape)
        return new_shapes

    def create_shape(self, slice_, value):
        assert isinstance(slice_[0], int)
        new_shape = []
        shape_offset = 0

        value_shape = (
            list(value.shape)
            if hasattr(value, "shape") and len(list(value.shape)) > 0
            else [1]
        )

        for i in range(1, len(self.shape)):

            if self.shape[i] is None:
                if i < len(slice_):
                    if isinstance(slice_[i], slice):
                        sl = slice_[i].stop
                        shape_offset += 1
                    else:
                        sl = slice_[i] + 1
                    new_shape.append(sl)
                else:
                    new_shape.append(value_shape[shape_offset])
                    shape_offset += 1
            elif i >= len(slice_) or isinstance(slice_[i], slice):
                shape_offset += 1
        new_appended_shape = list(self.shape)
        for i, dim in enumerate(self._dynamic_dims):
            new_appended_shape[dim] = new_shape[i]
        self._delete_chunks_after_reshape(slice_[0], new_appended_shape[1:])
        return new_shape

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

    def _delete_chunks_after_reshape(self, samples, new_shape: np.ndarray):
        """For a single sample or slice of samples deletes all chunks that exist out of new_shape bounds
        New shape does not include first (sample) dimension. It assumes that each sample gets the same new_shape shape
        NOTE: There is an assumption that dynamic_tensor chunks is either (1, A, B, C, ...) or (X, Infinity, Infinity, Infinity, ...)
        """
        if self.chunks[0] > 1:
            return

        if isinstance(samples, slice):
            samples_shape = self.get_shape([samples])
            for sample in range(slice.start, slice.stop, slice.step):
                self._delete_chunks_after_reshape_single_sample(
                    sample, samples_shape[sample], new_shape
                )
        else:
            assert isinstance(samples, int)
            self._delete_chunks_after_reshape_single_sample(
                samples, self.get_shape([samples]), new_shape
            )

    def _delete_chunks_after_reshape_single_sample(
        self, sample, sample_shape, new_shape
    ):
        if (sample_shape <= new_shape).all():
            return

        shapes = sample_shape
        assert len(shapes.shape) + 1 == len(self.shape)
        chunks = self._storage_tensor.chunks[1:]

        div = np.ceil(shapes / chunks).astype("int32")
        for index in np.ndindex(*div.tolist()):
            if (np.array(index) * chunks >= new_shape).any():
                try:
                    del self[".".join((sample,) + index)]
                except KeyError:
                    pass

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

    def commit(self):
        """ Deprecated alias to flush()"""
        self.flush()

    def flush(self):
        self._storage_tensor.store.flush()
        if self._dynamic_tensor:
            self._dynamic_tensor.store.flush()

    def close(self):
        self._storage_tensor.store.close()
        if self._dynamic_tensor:
            self._dynamic_tensor.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    @property
    def is_dynamic(self):
        return False if self._dynamic_tensor is None else True

    def disable_dynamicness(self):
        self._enabled_dynamicness = False

    def enable_dynamicness(self):
        self._enabled_dynamicness = True


def get_dynamic_dims(shape):
    return [i for i, s in enumerate(shape) if s is None]


def slice_stop_changed(slice_, new_stop):
    return slice(slice_.start, new_stop, slice_.step)
