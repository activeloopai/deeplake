import collections.abc as abc
import json
import math

import numpy as np
import zarr
import numcodecs

from hub.store.nested_store import NestedStore

from hub.exceptions import DynamicTensorNotFoundException, ValueShapeError
from hub.api.dataset_utils import slice_extract_info


def _tuple_product(tuple_):
    res = 1
    for t in tuple_:
        res *= t
    return res


def _determine_chunksizes(shape, dtype, block_size=2 ** 24):
    """
    Autochunking of tensors
    Chunk is determined by 16MB blocks keeping left dimensions inside a chunk
    Dimensions from left are kept until 16MB block is filled

    Parameters
    ----------
    shape: tuple
        the shape of the whole array
    dtype: type
        the type of the element (int, float)
    block_size: int (optional)
        how big the chunk size should be. Default to 16MB
    """

    sz = np.dtype(dtype).itemsize
    elem_count_in_chunk = block_size / sz

    # Get left most part which will be left static inside the chunk
    a = list(shape)
    a.reverse()
    left_part = shape
    prod = 1
    for i, dim in enumerate(a):
        prod *= dim
        if elem_count_in_chunk < prod:
            left_part = shape[-i:]
            break

    # If the tensor is smaller then the chunk size return
    if len(left_part) == len(shape):
        return list(left_part)

    # Get the middle chunk size of dimension
    els = math.ceil(elem_count_in_chunk / _tuple_product(left_part))

    # Contruct the chunksize shape
    chunksize = [els] + list(left_part)
    if len(chunksize) < len(shape):
        chunksize = [1] * (len(shape) - len(chunksize)) + chunksize
    return list(chunksize)


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
        compressor="default",
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
        exist_ = fs_map.get(".hub.dynamic_tensor")
        # if not exist_ and len(fs_map) > 0 and "w" in mode:
        #     raise OverwriteIsNotSafeException()
        exist = False if "w" in mode else exist_ is not None
        if "r" in mode and not exist:
            raise DynamicTensorNotFoundException()

        synchronizer = None
        # syncrhonizer = zarr.ProcessSynchronizer("~/activeloop/sync/example.sync")
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
                chunks=chunks or _determine_chunksizes(max_shape, dtype),
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
                )
                if self._dynamic_dims
                else None
            )
            fs_map[".hub.dynamic_tensor"] = bytes(json.dumps({"shape": shape}), "utf-8")
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
        if self._dynamic_tensor:
            self.set_shape(slice_, value)
        slice_ += [slice(0, None, 1) for i in self.max_shape[len(slice_) :]]
        real_shapes = self._dynamic_tensor[slice_[0]] if self._dynamic_tensor else None
        slice_ = self._get_slice(slice_, real_shapes)
        value = self.check_value_shape(value, slice_)
        self._storage_tensor[slice_] = value

    def check_value_shape(self, value, slice_):
        """Checks if value can be set to the slice"""
        if None not in self.shape and self.dtype != 'O':
            if not all([isinstance(sh, int) for sh in slice_]):
                expected_value_shape = tuple(
                    [
                        len(range(*slice_shape.indices(self.shape[i])))
                        for i, slice_shape in enumerate(slice_)
                        if not isinstance(slice_shape, int)
                    ]
                )                
                if expected_value_shape[0] == 1 and len(expected_value_shape) > 1:
                    expected_value_shape = expected_value_shape[1:]

                if isinstance(value, list):
                    value = np.array(value)
                if isinstance(value, np.ndarray):
                    if value.shape[0] == 1 and expected_value_shape[0] != 1:
                        value = np.squeeze(value, axis=0)
                    if value.shape[-1] == 1 and expected_value_shape[-1] != 1:
                        value = np.squeeze(value, axis=-1)
                    if value.shape != expected_value_shape:
                        raise ValueShapeError(expected_value_shape, value.shape)
            else:
                expected_value_shape = (1,)
                if isinstance(value, list):
                    value = np.array(value)                
                if isinstance(value, np.ndarray) and value.shape != expected_value_shape:                    
                    raise ValueShapeError(expected_value_shape, value.shape)
        return value

    def get_shape(self, slice_):
        """Gets the shape of the slice from tensor"""
        if self._dynamic_tensor is None:
            return self.shape
        if isinstance(slice_[0], slice):
            num, ofs = slice_extract_info(slice_[0], self.shape[0])
            slice_[0] = ofs if num == 1 else slice_[0]
        if isinstance(slice_[0], int):
            final_shape = []
            shape_offset = 0
            for i in range(1, len(self.shape)):
                if i < len(slice_):
                    if isinstance(slice_[i], slice):
                        final_shape.append(slice_[i].stop - slice_[i].start)
                    shape_offset += 1
                elif self.shape[i] is not None:
                    final_shape.append(self.shape[i])
                elif shape_offset < len(self._dynamic_tensor[slice_[0]]):
                    final_shape.append(self._dynamic_tensor[slice_[0]][shape_offset])
                    shape_offset += 1
            return tuple(final_shape)
        else:
            raise ValueError("Getting shape across multiple dimensions isn't supported")

    def set_shape(self, slice_, value):
        """Sets the shape of the slice of tensor"""
        if isinstance(slice_[0], slice):
            num, ofs = slice_extract_info(slice_[0], self.shape[0])
            slice_[0] = ofs if num == 1 else slice_[0]
        if isinstance(slice_[0], int):
            value_shape = list(value.shape) if isinstance(value, np.ndarray) else [1]
            new_shape = []
            shape_offset = 0
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
                elif i < len(slice_) and isinstance(slice_[i], slice):
                    shape_offset += 1
            self._dynamic_tensor[slice_[0]] = np.maximum(
                self._dynamic_tensor[slice_[0]], new_shape
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
        self._storage_tensor.store.commit()
        if self._dynamic_tensor:
            self._dynamic_tensor.store.commit()


def get_dynamic_dims(shape):
    return [i for i, s in enumerate(shape) if s is None]


def slice_stop_changed(slice_, new_stop):
    return slice(slice_.start, new_stop, slice_.step)
