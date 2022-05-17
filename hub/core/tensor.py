import hub
from hub.core.linked_chunk_engine import LinkedChunkEngine
from hub.core.storage.lru_cache import LRUCache
from hub.util.invalid_view_op import invalid_view_op
from hub.core.version_control.commit_chunk_set import CommitChunkSet
from hub.core.version_control.commit_diff import CommitDiff
from hub.core.chunk.base_chunk import InputSample
import numpy as np
from typing import Dict, List, Sequence, Union, Optional, Tuple, Any, Callable
from functools import reduce, partial
from hub.core.index import Index, IndexEntry
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.storage import StorageProvider
from hub.core.chunk_engine import ChunkEngine
from hub.core.tensor_link import get_link_transform
from hub.api.info import load_info
from hub.util.keys import (
    get_chunk_id_encoder_key,
    get_chunk_key,
    get_tensor_commit_chunk_set_key,
    get_tensor_commit_diff_key,
    get_tensor_meta_key,
    get_tensor_tile_encoder_key,
    get_sequence_encoder_key,
    tensor_exists,
    get_tensor_info_key,
    get_sample_id_tensor_key,
)
from hub.util.keys import (
    get_tensor_meta_key,
    tensor_exists,
    get_tensor_info_key,
    get_sample_id_tensor_key,
    get_sample_info_tensor_key,
    get_sample_shape_tensor_key,
)
from hub.util.modified import get_modified_indexes
from hub.util.shape_interval import ShapeInterval
from hub.util.exceptions import (
    TensorDoesNotExistError,
    InvalidKeyTypeError,
    TensorAlreadyExistsError,
)

from hub.util.pretty_print import (
    max_array_length,
    get_string,
    summary_tensor,
)
from hub.constants import FIRST_COMMIT_ID, MB, _NO_LINK_UPDATE


from hub.util.version_control import auto_checkout

from hub.compression import get_compression_type, VIDEO_COMPRESSION
from hub.util.notebook import is_jupyter, video_html, is_colab
import warnings
import webbrowser


def create_tensor(
    key: str,
    storage: StorageProvider,
    htype: str,
    sample_compression: str,
    chunk_compression: str,
    version_state: Dict[str, Any],
    **kwargs,
):
    """If a tensor does not exist, create a new one with the provided meta.

    Args:
        key (str): Key for where the chunks, index_meta, and tensor_meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider that all tensor data is written to.
        htype (str): Htype is how the default tensor metadata is defined.
        sample_compression (str): All samples will be compressed in the provided format. If `None`, samples are uncompressed.
        chunk_compression (str): All chunks will be compressed in the provided format. If `None`, chunks are uncompressed.
        version_state (Dict[str, Any]): The version state of the dataset, includes commit_id, commit_node, branch, branch_commit_map and commit_node_map.
        **kwargs: `htype` defaults can be overridden by passing any of the compatible parameters.
            To see all `htype`s and their correspondent arguments, check out `hub/htypes.py`.

    Raises:
        TensorAlreadyExistsError: If a tensor defined with `key` already exists.
    """

    commit_id = version_state["commit_id"]
    if tensor_exists(key, storage, commit_id):
        raise TensorAlreadyExistsError(key)

    meta_key = get_tensor_meta_key(key, commit_id)
    meta = TensorMeta(
        htype=htype,
        sample_compression=sample_compression,
        chunk_compression=chunk_compression,
        **kwargs,
    )
    storage[meta_key] = meta  # type: ignore

    if commit_id != FIRST_COMMIT_ID:
        cset_key = get_tensor_commit_chunk_set_key(key, commit_id)
        cset = CommitChunkSet()
        storage[cset_key] = cset  # type: ignore

    diff_key = get_tensor_commit_diff_key(key, commit_id)
    diff = CommitDiff(created=True)
    storage[diff_key] = diff  # type: ignore


def delete_tensor(key: str, dataset):
    """Delete tensor from storage.

    Args:
        key (str): Key for where the chunks, index_meta, and tensor_meta will be located in `storage` relative to it's root.
        dataset (Dataset): Dataset that the tensor is located in.

    Raises:
        TensorDoesNotExistError: If no tensor with `key` exists and a `tensor_meta` was not provided.
    """
    storage = dataset.storage
    version_state = dataset.version_state
    tensor = Tensor(key, dataset)
    chunk_engine: ChunkEngine = tensor.chunk_engine
    enc = chunk_engine.chunk_id_encoder
    n_chunks = chunk_engine.num_chunks
    chunk_names = [enc.get_name_for_chunk(i) for i in range(n_chunks)]
    chunk_keys = [
        get_chunk_key(key, chunk_name, version_state["commit_id"])
        for chunk_name in chunk_names
    ]
    for chunk_key in chunk_keys:
        try:
            del storage[chunk_key]
        except KeyError:
            pass

    commit_id = version_state["commit_id"]
    meta_key = get_tensor_meta_key(key, commit_id)
    try:
        del storage[meta_key]
    except KeyError:
        pass

    info_key = get_tensor_info_key(key, commit_id)
    try:
        del storage[info_key]
    except KeyError:
        pass

    diff_key = get_tensor_commit_diff_key(key, commit_id)
    try:
        del storage[diff_key]
    except KeyError:
        pass

    chunk_id_encoder_key = get_chunk_id_encoder_key(key, commit_id)
    try:
        del storage[chunk_id_encoder_key]
    except KeyError:
        pass

    tile_encoder_key = get_tensor_tile_encoder_key(key, commit_id)
    try:
        del storage[tile_encoder_key]
    except KeyError:
        pass

    seq_encoder_key = get_sequence_encoder_key(key, commit_id)
    try:
        del storage[seq_encoder_key]
    except KeyError:
        pass


def _inplace_op(f):
    op = f.__name__

    def inner(tensor, other):
        tensor._write_initialization()
        tensor.chunk_engine.update(
            tensor.index,
            other,
            op,
            link_callback=tensor._update_links if tensor.meta.links else None,
        )
        if not tensor.index.is_trivial():
            tensor._skip_next_setitem = True
        return tensor

    return inner


class Tensor:
    def __init__(
        self,
        key: str,
        dataset,
        index: Optional[Index] = None,
        is_iteration: bool = False,
        chunk_engine: Optional[ChunkEngine] = None,
    ):
        """Initializes a new tensor.

        Note:
            This operation does not create a new tensor in the storage provider,
            and should normally only be performed by Hub internals.

        Args:
            key (str): The internal identifier for this tensor.
            dataset (Dataset): The dataset that this tensor is located in.
            index: The Index object restricting the view of this tensor.
                Can be an int, slice, or (used internally) an Index object.
            is_iteration (bool): If this tensor is being used as an iterator.
            chunk_engine (ChunkEngine, optional): The underlying chunk_engine for the tensor

        Raises:
            TensorDoesNotExistError: If no tensor with `key` exists and a `tensor_meta` was not provided.
        """
        self.key = key
        self.dataset = dataset
        self.storage: LRUCache = dataset.storage
        self.index = index or Index()
        self.version_state = dataset.version_state
        self.link_creds = dataset.link_creds
        self.is_iteration = is_iteration
        commit_id = self.version_state["commit_id"]

        if not self.is_iteration and not tensor_exists(
            self.key, self.storage, commit_id
        ):
            raise TensorDoesNotExistError(self.key)

        meta_key = get_tensor_meta_key(self.key, commit_id)
        meta = self.storage.get_hub_object(meta_key, TensorMeta)
        if chunk_engine is not None:
            self.chunk_engine = chunk_engine
        elif meta.is_link:
            self.chunk_engine = LinkedChunkEngine(
                self.key,
                self.storage,
                self.version_state,
                link_creds=dataset.link_creds,
            )
        else:
            self.chunk_engine = ChunkEngine(self.key, self.storage, self.version_state)

        if not self.is_iteration:
            self.index.validate(self.num_samples)

        # An optimization to skip multiple .numpy() calls when performing inplace ops on slices:
        self._skip_next_setitem = False

    def _write_initialization(self):
        self.storage.check_readonly()
        # if not the head node, checkout to an auto branch that is newly created
        if auto_checkout(self.dataset):
            self.chunk_engine = self.version_state["full_tensors"][
                self.key
            ].chunk_engine

    @invalid_view_op
    def extend(
        self,
        samples: Union[np.ndarray, Sequence[InputSample], "Tensor"],
        progressbar: bool = False,
    ):

        """Extends the end of the tensor by appending multiple elements from a sequence. Accepts a sequence, a single batched numpy array,
        or a sequence of `hub.read` outputs, which can be used to load files. See examples down below.

        Example:
            Numpy input:

            >>> len(tensor)
            0
            >>> tensor.extend(np.zeros((100, 28, 28, 1)))
            >>> len(tensor)
            100


            File input:

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
            progressbar (bool): Specifies whether a progressbar should be displayed while extending.

        Raises:
            TensorDtypeMismatchError: TensorDtypeMismatchError: Dtype for array must be equal to or castable to this tensor's dtype
        """
        self.check_link_ready()
        self._write_initialization()
        self.chunk_engine.extend(
            samples,
            progressbar=progressbar,
            link_callback=self._append_to_links if self.meta.links else None,
        )

    @property
    def info(self):
        """Returns the information about the tensor.

        Returns:
            TensorInfo: Information about the tensor.
        """
        commit_id = self.version_state["commit_id"]
        chunk_engine = self.chunk_engine
        if chunk_engine._info is None or chunk_engine._info_commit_id != commit_id:
            path = get_tensor_info_key(self.key, commit_id)
            chunk_engine._info = load_info(path, self.dataset, self.key)
            chunk_engine._info_commit_id = commit_id
            self.storage.register_hub_object(path, chunk_engine._info)
        return chunk_engine._info

    @info.setter
    def info(self, value):
        if isinstance(value, dict):
            info = self.info
            info.replace_with(value)
        else:
            raise TypeError("Info must be set with type Dict")

    @invalid_view_op
    def append(self, sample: InputSample):
        """Appends a single sample to the end of the tensor. Can be an array, scalar value, or the return value from `hub.read`,
        which can be used to load files. See examples down below.

        Examples:
            Numpy input:

            >>> len(tensor)
            0
            >>> tensor.append(np.zeros((28, 28, 1)))
            >>> len(tensor)
            1

            File input:

            >>> len(tensor)
            0
            >>> tensor.append(hub.read("path/to/file"))
            >>> len(tensor)
            1

        Args:
            sample (InputSample): The data to append to the tensor. `Sample` is generated by `hub.read`. See the above examples.
        """
        self.extend([sample], progressbar=False)

    def clear(self):
        """Deletes all samples from the tensor"""
        self.chunk_engine.clear()
        sample_id_key = get_sample_id_tensor_key(self.key)
        try:
            sample_id_tensor = Tensor(sample_id_key, self.dataset)
            sample_id_tensor.chunk_engine.clear()
            self.meta.links.clear()
            self.meta.is_dirty = True
        except TensorDoesNotExistError:
            pass

    def modified_samples(
        self, target_id: Optional[str] = None, return_indexes: Optional[bool] = False
    ):
        """Returns a slice of the tensor with only those elements that were modified/added.
        By default the modifications are calculated relative to the previous commit made, but this can be changed by providing a `target id`.

        Args:
            target_id (str, optional): The commit id or branch name to calculate the modifications relative to. Defaults to None.
            return_indexes (bool, optional): If True, returns the indexes of the modified elements. Defaults to False.

        Returns:
            Tensor: A new tensor with only the modified elements if `return_indexes` is False.
            Tuple[Tensor, List[int]]: A new tensor with only the modified elements and the indexes of the modified elements if `return_indexes` is True.

        Raises:
            TensorModifiedError: If a target id is passed which is not an ancestor of the current commit.
        """
        current_commit_id = self.version_state["commit_id"]
        indexes = get_modified_indexes(
            self.key,
            current_commit_id,
            target_id,
            self.version_state,
            self.storage,
        )
        tensor = self[indexes]
        if return_indexes:
            return tensor, indexes
        return tensor

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
        sample_shape_tensor = self._sample_shape_tensor
        sample_shape_provider = (
            self._sample_shape_provider(sample_shape_tensor)
            if sample_shape_tensor
            else None
        )
        if sample_shape_provider is None:
            self.check_link_ready()
        shape: Tuple[Optional[int], ...]
        shape = self.chunk_engine.shape(
            self.index, sample_shape_provider=sample_shape_provider
        )
        if not shape and self.meta.max_shape:
            shape = (0,) * len(self.meta.max_shape)
        if self.meta.max_shape == [0, 0, 0]:
            shape = ()
        return shape

    @property
    def size(self) -> Optional[int]:
        s = 1
        for x in self.shape:
            if x is None:
                return None
            s *= x  # not using np.prod to avoid overflow
        return s

    @property
    def ndim(self) -> int:
        return self.chunk_engine.ndim(self.index)

    @property
    def dtype(self) -> Optional[np.dtype]:
        if self.base_htype in ("json", "list"):
            return np.dtype(str)
        if self.meta.dtype:
            return np.dtype(self.meta.dtype)
        return None

    @property
    def is_sequence(self):
        return self.meta.is_sequence

    @property
    def is_link(self):
        return self.meta.is_link

    @property
    def verify(self):
        return self.is_link and self.meta.verify

    @property
    def htype(self):
        htype = self.meta.htype
        if self.is_sequence:
            htype = f"sequence[{htype}]"
        if self.is_link:
            htype = f"link[{htype}]"
        return htype

    @property
    def hidden(self) -> bool:
        return self.meta.hidden

    @property
    def base_htype(self):
        return self.meta.htype

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
        return self.chunk_engine.shape_interval

    @property
    def is_dynamic(self) -> bool:
        """Will return True if samples in this tensor have shapes that are unequal."""
        return self.shape_interval.is_dynamic

    @property
    def num_samples(self) -> int:
        """Returns the length of the primary axis of the tensor.
        Ignores any applied indexing and returns the total length.
        """
        if self.is_sequence:
            return self.chunk_engine._sequence_length
        return self.meta.length

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

        return self.index.length(self.num_samples)

    def __getitem__(
        self,
        item: Union[int, slice, List[int], Tuple[Union[int, slice, Tuple[int]]], Index],
        is_iteration: bool = False,
    ):
        if not isinstance(item, (int, slice, list, tuple, Index)):
            raise InvalidKeyTypeError(item)
        return Tensor(
            self.key,
            self.dataset,
            index=self.index[item],
            is_iteration=is_iteration,
            chunk_engine=self.chunk_engine,
        )

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
            return np.dtype(bool)
        elif isinstance(val, Sequence):
            return reduce(self._get_bigger_dtype, map(self._infer_np_dtype, val))
        else:
            raise TypeError(f"Cannot infer numpy dtype for {val}")

    def __setitem__(self, item: Union[int, slice], value: Any):
        """Update samples with new values.

        Example:
            >>> tensor.append(np.zeros((10, 10)))
            >>> tensor.shape
            (1, 10, 10)
            >>> tensor[0] = np.zeros((3, 3))
            >>> tensor.shape
            (1, 3, 3)
        """
        self.check_link_ready()
        self._write_initialization()
        update_link_callback = self._update_links if self.meta.links else None
        if isinstance(value, Tensor):
            if value._skip_next_setitem:
                value._skip_next_setitem = False
                return
            value = value.numpy(aslist=True)
        item_index = Index(item)

        if (
            hub.constants._ENABLE_RANDOM_ASSIGNMENT
            and isinstance(item, int)
            and item >= self.num_samples
        ):
            num_samples_to_pad = item - self.num_samples
            append_link_callback = self._append_to_links if self.meta.links else None

            self.chunk_engine.pad_and_append(
                num_samples_to_pad,
                value,
                append_link_callback=append_link_callback,
                update_link_callback=update_link_callback,
            )
            return

        self.chunk_engine.update(
            self.index[item_index],
            value,
            link_callback=update_link_callback,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i, is_iteration=True)

    def check_link_ready(self):
        if not self.is_link:
            return
        missing_keys = self.link_creds.missing_keys
        if missing_keys:
            raise ValueError(
                f"Not all credentials are populated for a tensor with linked data. Missing: {missing_keys}. Populate with `dataset.populate_creds(key, value)`."
            )

    def numpy(
        self, aslist=False, fetch_chunks=False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Computes the contents of the tensor in numpy format.

        Args:
            aslist (bool): If True, a list of np.ndarrays will be returned. Helpful for dynamic tensors.
                If False, a single np.ndarray will be returned unless the samples are dynamically shaped, in which case
                an error is raised.
            fetch_chunks (bool): If True, full chunks will be retrieved from the storage, otherwise only required bytes will be retrieved.
                This will always be True even if specified as False in the following cases:
                - The tensor is ChunkCompressed
                - The chunk which is being accessed has more than 128 samples.

        Raises:
            DynamicTensorNumpyError: If reading a dynamically-shaped array slice without `aslist=True`.
            ValueError: If the tensor is a link and the credentials are not populated.

        Returns:
            A numpy array containing the data represented by this tensor.
        """
        self.check_link_ready()
        return self.chunk_engine.numpy(
            self.index, aslist=aslist, fetch_chunks=fetch_chunks
        )

    def summary(self):
        pretty_print = summary_tensor(self)

        print(self)
        print(pretty_print)

    def __str__(self):
        index_str = f", index={self.index}"
        if self.index.is_trivial():
            index_str = ""
        return f"Tensor(key={repr(self.meta.name or self.key)}{index_str})"

    __repr__ = __str__

    def __array__(self) -> np.ndarray:
        return self.numpy()  # type: ignore

    @_inplace_op
    def __iadd__(self, other):
        pass

    @_inplace_op
    def __isub__(self, other):
        pass

    @_inplace_op
    def __imul__(self, other):
        pass

    @_inplace_op
    def __itruediv__(self, other):
        pass

    @_inplace_op
    def __ifloordiv__(self, other):
        pass

    @_inplace_op
    def __imod__(self, other):
        pass

    @_inplace_op
    def __ipow__(self, other):
        pass

    @_inplace_op
    def __ilshift__(self, other):
        pass

    @_inplace_op
    def __irshift__(self, other):
        pass

    @_inplace_op
    def __iand__(self, other):
        pass

    @_inplace_op
    def __ixor__(self, other):
        pass

    @_inplace_op
    def __ior__(self, other):
        pass

    def data(self, aslist: bool = False) -> Any:
        htype = self.base_htype
        if htype in ("json", "text"):

            if self.ndim == 1:
                return self.numpy()[0]
            else:
                return [sample[0] for sample in self.numpy(aslist=True)]
        elif htype == "list":
            if self.ndim == 1:
                return list(self.numpy())
            else:
                return list(map(list, self.numpy(aslist=True)))
        else:
            return self.numpy(aslist=aslist)

    def tobytes(self) -> bytes:
        """Returns the bytes of the tensor.

        - Only works for a single sample of tensor.
        - If the tensor is uncompressed, this returns the bytes of the numpy array.
        - If the tensor is sample compressed, this returns the compressed bytes of the sample.
        - If the tensor is chunk compressed, this raises an error.

        Returns:
            bytes: The bytes of the tensor.

        Raises:
            ValueError: If the tensor has multiple samples.
        """
        if self.index.values[0].subscriptable() or len(self.index.values) > 1:
            raise ValueError("tobytes() can be used only on exatcly 1 sample.")
        self.check_link_ready()
        idx = self.index.values[0].value
        if self.is_link:
            return self.chunk_engine.get_hub_read_sample(idx).compressed_bytes  # type: ignore
        return self.chunk_engine.read_bytes_for_sample(idx)  # type: ignore

    def _pop(self):
        self.chunk_engine._pop()
        [self.dataset[link]._pop() for link in self.meta.links]

    def _append_to_links(self, sample, flat: Optional[bool]):
        for k, v in self.meta.links.items():
            if flat is None or v["flatten_sequence"] == flat:
                v = get_link_transform(v["append"])(sample, self.link_creds)
                tensor = Tensor(k, self.dataset)
                if (
                    isinstance(v, np.ndarray)
                    and tensor.dtype
                    and v.dtype != tensor.dtype
                ):
                    v = v.astype(tensor.dtype)  # bc
                tensor.append(v)

    def _update_links(
        self,
        global_sample_index: int,
        sub_index: Index,
        new_sample,
        flat: Optional[bool],
    ):
        for k, v in self.meta.links.items():
            if flat is None or v["flatten_sequence"] == flat:
                fname = v.get("update")
                if fname:
                    func = get_link_transform(fname)
                    tensor = Tensor(k, self.dataset)
                    val = func(
                        new_sample,
                        tensor[global_sample_index],
                        sub_index=sub_index,
                        partial=not sub_index.is_trivial(),
                        link_creds=self.link_creds,
                    )
                    if val is not _NO_LINK_UPDATE:
                        if (
                            isinstance(val, np.ndarray)
                            and tensor.dtype
                            and val.dtype != tensor.dtype
                        ):
                            val = val.astype(tensor.dtype)  # bc
                        tensor[global_sample_index] = val

    @property
    def _sample_info_tensor(self):
        return self.dataset._tensors().get(get_sample_info_tensor_key(self.key))

    @property
    def _sample_shape_tensor(self):
        return self.dataset._tensors().get(get_sample_shape_tensor_key(self.key))

    @property
    def _sample_id_tensor(self):
        return self.dataset._tensors().get(get_sample_id_tensor_key(self.key))

    def _sample_shape_provider(self, sample_shape_tensor) -> Callable:
        if self.is_sequence:

            def get_sample_shape(global_sample_index: int):
                seq_pos = slice(
                    *self.chunk_engine.sequence_encoder[global_sample_index]
                )
                idx = Index([IndexEntry(seq_pos)])
                shapes = sample_shape_tensor[idx].numpy()
                return shapes

        else:

            def get_sample_shape(global_sample_index: int):
                return tuple(sample_shape_tensor[global_sample_index].numpy().tolist())

        return get_sample_shape

    def _get_sample_info_at_index(self, global_sample_index: int, sample_info_tensor):
        if self.is_sequence:
            return [
                sample_info_tensor[i].data()
                for i in range(*self.chunk_engine.sequence_encoder[global_sample_index])
            ]
        return sample_info_tensor[global_sample_index].data()

    def _sample_info(self, index: Index):
        sample_info_tensor = self._sample_info_tensor
        if sample_info_tensor is None:
            return None
        if index.subscriptable_at(0):
            return list(
                map(
                    partial(
                        self._get_sample_info_at_index,
                        sample_info_tensor=sample_info_tensor,
                    ),
                    index.values[0].indices(self.num_samples),
                )
            )
        return self._get_sample_info_at_index(index.values[0].value, sample_info_tensor)  # type: ignore

    @property
    def sample_info(self):
        return self._sample_info(self.index)

    def _linked_sample(self):
        if not self.is_link:
            raise ValueError("Not supported as the tensor is not a link.")
        if self.index.values[0].subscriptable() or len(self.index.values) > 1:
            raise ValueError("_linked_sample can be used only on exatcly 1 sample.")
        return self.chunk_engine.linked_sample(self.index.values[0].value)

    def _get_video_stream_url(self):
        from hub.visualizer.video_streaming import get_video_stream_url

        return get_video_stream_url(self, self.index.values[0].value)

    def play(self):
        if get_compression_type(self.meta.sample_compression) != VIDEO_COMPRESSION:
            raise Exception("Only supported for video tensors.")
        if self.index.values[0].subscriptable():
            raise ValueError("Video streaming requires exactly 1 sample.")
        if len(self.index.values) > 1:
            warnings.warn(
                "Sub indexes to video sample will be ignored while streaming."
            )
        if is_colab():
            raise NotImplementedError("Video streaming is not supported on colab yet.")
        elif is_jupyter():
            return video_html(
                src=self._get_video_stream_url(),
                alt=f"{self.key}[{self.index.values[0].value}]",
            )
        else:
            webbrowser.open(self._get_video_stream_url())
