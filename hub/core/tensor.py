import numpy as np
from typing import List, Sequence, Union, Optional, Tuple, Any
from functools import reduce
from hub.core.index import Index
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.storage import StorageProvider, LRUCache
from hub.core.sample import Sample, SampleValue  # type: ignore
from hub.core.chunk_engine import ChunkEngine
from hub.api.info import load_info
from hub.util.keys import get_tensor_meta_key, tensor_exists, get_tensor_info_key
from hub.util.casting import get_incompatible_dtype
from hub.util.shape_interval import ShapeInterval
from hub.util.exceptions import (
    TensorDoesNotExistError,
    InvalidKeyTypeError,
    TensorAlreadyExistsError,
    TensorDtypeMismatchError,
)


def create_tensor(
    key: str,
    storage: StorageProvider,
    htype: str,
    sample_compression: str,
    **kwargs,
):
    """If a tensor does not exist, create a new one with the provided meta.

    Args:
        key (str): Key for where the chunks, index_meta, and tensor_meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider that all tensor data is written to.
        htype (str): Htype is how the default tensor metadata is defined.
        sample_compression (str): All samples will be compressed in the provided format. If `None`, samples are uncompressed.
        **kwargs: `htype` defaults can be overridden by passing any of the compatible parameters.
            To see all `htype`s and their correspondent arguments, check out `hub/htypes.py`.

    Raises:
        TensorAlreadyExistsError: If a tensor defined with `key` already exists.
    """

    if tensor_exists(key, storage):
        raise TensorAlreadyExistsError(key)

    meta_key = get_tensor_meta_key(key)
    meta = TensorMeta(
        htype=htype,
        sample_compression=sample_compression,
        **kwargs,
    )
    storage[meta_key] = meta  # type: ignore


class Tensor:
    def __init__(
        self,
        key: str,
        storage: LRUCache,
        index: Optional[Index] = None,
    ):
        """Initializes a new tensor.

        Note:
            This operation does not create a new tensor in the storage provider,
            and should normally only be performed by Hub internals.

        Args:
            key (str): The internal identifier for this tensor.
            storage (LRUCache): The storage provider for the parent dataset.
            index: The Index object restricting the view of this tensor.
                Can be an int, slice, or (used internally) an Index object.

        Raises:
            TensorDoesNotExistError: If no tensor with `key` exists and a `tensor_meta` was not provided.
        """

        self.key = key
        self.storage = storage
        self.index = index or Index()

        if not tensor_exists(self.key, self.storage):
            raise TensorDoesNotExistError(self.key)

        self.chunk_engine = ChunkEngine(self.key, self.storage)
        self.index.validate(self.num_samples)
        self.info = load_info(get_tensor_info_key(self.key), self.storage)

    def extend(self, samples: Union[np.ndarray, Sequence[SampleValue]]):
        """Extends the end of the tensor by appending multiple elements from a sequence. Accepts a sequence, a single batched numpy array,
        or a sequence of `hub.read` outputs, which can be used to load files. See examples down below.

        Example:
            numpy input:
                >>> len(tensor)
                0
                >>> tensor.extend(np.zeros((100, 28, 28, 1)))
                >>> len(tensor)
                100

            file input:
                >>> len(tensor)
                0
                >>> tensor.extend([
                        hub.read("path/to/image1"),
                        hub.read("path/to/image2"),
                    ])
                >>> len(tensor)
                2


        Args:
            samples (np.ndarray, Sequence, Sequence[Sample]): The data to add to the tensor.
                The length should be equal to the number of samples to add.

        Raises:
            TensorDtypeMismatchError: TensorDtypeMismatchError: Dtype for array must be equal to or castable to this tensor's dtype
        """

        self.chunk_engine.extend(samples)

    def append(
        self,
        sample: Union[np.ndarray, float, int, Sample],
    ):
        """Appends a single sample to the end of the tensor. Can be an array, scalar value, or the return value from `hub.read`,
        which can be used to load files. See examples down below.

        Examples:
            numpy input:
                >>> len(tensor)
                0
                >>> tensor.append(np.zeros((28, 28, 1)))
                >>> len(tensor)
                1

            file input:
                >>> len(tensor)
                0
                >>> tensor.append(hub.read("path/to/file"))
                >>> len(tensor)
                1

        Args:
            sample (np.ndarray, float, int, Sample): The data to append to the tensor. `Sample` is generated by `hub.read`. See the above examples.
        """
        self.extend([sample])

    @property
    def meta(self):
        return self.chunk_engine.tensor_meta

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        """Get the shape of this tensor. Length is included.

        Note:
            If you don't want `None` in the output shape or want the lower/upper bound shapes,
            use `tensor.shape_interval` instead.

        Example:
            >>> tensor.append(np.zeros((10, 10)))
            >>> tensor.append(np.zeros((10, 15)))
            >>> tensor.shape
            (2, 10, None)

        Returns:
            tuple: Tuple where each value is either `None` (if that axis is dynamic) or
                an `int` (if that axis is fixed).
        """

        return self.shape_interval.astuple()

    @property
    def dtype(self) -> np.dtype:
        if self.meta.dtype:
            return np.dtype(self.meta.dtype)
        return None

    @property
    def shape_interval(self) -> ShapeInterval:
        """Returns a `ShapeInterval` object that describes this tensor's shape more accurately. Length is included.

        Note:
            If you are expecting a `tuple`, use `tensor.shape` instead.

        Example:
            >>> tensor.append(np.zeros((10, 10)))
            >>> tensor.append(np.zeros((10, 15)))
            >>> tensor.shape_interval
            ShapeInterval(lower=(2, 10, 10), upper=(2, 10, 15))
            >>> str(tensor.shape_interval)
            (2, 10, 10:15)

        Returns:
            ShapeInterval: Object containing `lower` and `upper` properties.
        """

        length = [len(self)]

        min_shape = length + list(self.meta.min_shape)
        max_shape = length + list(self.meta.max_shape)

        return ShapeInterval(min_shape, max_shape)

    @property
    def is_dynamic(self) -> bool:
        """Will return True if samples in this tensor have shapes that are unequal."""
        return self.shape_interval.is_dynamic

    @property
    def num_samples(self) -> int:
        """Returns the length of the primary axis of the tensor.
        Ignores any applied indexing and returns the total length.
        """
        return self.chunk_engine.num_samples

    def __len__(self):
        """Returns the length of the primary axis of the tensor.
        Accounts for indexing into the tensor object.

        Examples:
            >>> len(tensor)
            0
            >>> tensor.extend(np.zeros((100, 10, 10)))
            >>> len(tensor)
            100
            >>> len(tensor[5:10])
            5

        Returns:
            int: The current length of this tensor.
        """

        # catch corrupted datasets / user tampering ASAP
        self.chunk_engine.validate_num_samples_is_synchronized()

        return self.index.length(self.meta.length)

    def __getitem__(
        self,
        item: Union[int, slice, List[int], Tuple[Union[int, slice, Tuple[int]]], Index],
    ):
        if not isinstance(item, (int, slice, list, tuple, Index)):
            raise InvalidKeyTypeError(item)
        return Tensor(self.key, self.storage, index=self.index[item])

    def _get_bigger_dtype(self, d1, d2):
        if np.can_cast(d1, d2):
            if np.can_cast(d2, d1):
                return d1
            else:
                return d2
        else:
            if np.can_cast(d2, d1):
                return d2
            else:
                return np.object

    def _infer_np_dtype(self, val: Any) -> np.dtype:
        # TODO refac
        if hasattr(val, "dtype"):
            return val.dtype
        elif isinstance(val, int):
            return np.array(0).dtype
        elif isinstance(val, float):
            return np.array(0.0).dtype
        elif isinstance(val, str):
            return np.array("").dtype
        elif isinstance(val, bool):
            return np.bool
        elif isinstance(val, Sequence):
            return reduce(self._get_bigger_dtype, map(self._infer_np_dtype, val))
        else:
            raise TypeError(f"Cannot infer numpy dtype for {val}")

    def __setitem__(self, item: Union[int, slice], value: Any):
        """Update samples with new values. Sub-slice updates are not supported yet.

        Example:
            >>> tensor.append(np.zeros((10, 10)))
            >>> tensor.shape
            (1, 10, 10)
            >>> tensor[0] = np.zeros((3, 3))
            >>> tensor.shape
            (1, 3, 3)
        """

        item_index = Index(item)
        self.chunk_engine.update(self.index[item_index], value)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def numpy(self, aslist=False) -> Union[np.ndarray, List[np.ndarray]]:
        """Computes the contents of the tensor in numpy format.

        Args:
            aslist (bool): If True, a list of np.ndarrays will be returned. Helpful for dynamic tensors.
                If False, a single np.ndarray will be returned unless the samples are dynamically shaped, in which case
                an error is raised.

        Raises:
            DynamicTensorNumpyError: If reading a dynamically-shaped array slice without `aslist=True`.

        Returns:
            A numpy array containing the data represented by this tensor.
        """

        return self.chunk_engine.numpy(self.index, aslist=aslist)

    def __str__(self):
        index_str = f", index={self.index}"
        if self.index.is_trivial():
            index_str = ""
        return f"Tensor(key={repr(self.key)}{index_str})"

    __repr__ = __str__

    def __array__(self) -> np.ndarray:
        return self.numpy()

    # TODO: Inplace operatitions
    def __iadd__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __idiv__(self, other):
        raise NotImplementedError

    def __ifloordiv__(self, other):
        raise NotImplementedError

    def __imod__(self, other):
        raise NotImplementedError

    def __ipow__(self, other):
        raise NotImplementedError

    def __ilshift__(self, other):
        raise NotImplementedError

    def __irshift__(self, other):
        raise NotImplementedError

    def __iand__(self, other):
        raise NotImplementedError

    def __ixor__(self, other):
        raise NotImplementedError

    def __ior__(self, other):
        raise NotImplementedError
