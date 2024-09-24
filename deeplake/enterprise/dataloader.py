from typing import Callable, Dict, List, Optional, Union
import deeplake
from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake
from deeplake.enterprise.dummy_dataloader import DummyDataloader  # type: ignore
from deeplake.util.scheduling import create_fetching_schedule, find_primary_tensor
from deeplake.core.seed import DeeplakeRandom
from deeplake.enterprise.util import (
    handle_mode,
    raise_indra_installation_error,
    verify_base_storage,
    get_collate_fn,
)
from deeplake.hooks import dataset_read
from deeplake.enterprise.libdeeplake_query import query, sample_by
from deeplake.integrations.pytorch.common import (
    PytorchTransformFunction,
    check_tensors,
    validate_decode_method,
    find_additional_tensors_and_info,
    get_htype_ndim_tensor_info_dicts,
)
from deeplake.util.dataset import map_tensor_keys
from functools import partial
import importlib

try:
    from torch.utils.data.dataloader import DataLoader, _InfiniteConstantSampler
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import BatchSampler

except ImportError:
    DataLoader = object  # type: ignore
    _InfiniteConstantSampler = None  # type: ignore
    DistributedSampler = None  # type: ignore
    BatchSampler = None  # type: ignore

import numpy as np
import warnings

import math

import itertools
import multiprocessing

original_islice = itertools.islice


def deeplake_islice(iterable, *args, **kwargs):
    if isinstance(iterable, DeepLakeDataLoader):
        return iter(iterable)
    return original_islice(iterable, *args, **kwargs)


itertools.islice = deeplake_islice  # type: ignore


# Load lazy to avoid cycylic import.
INDRA_LOADER = None  # type: ignore


def indra_available() -> bool:
    try:
        import_indra_loader()
        return True
    except ImportError:
        return False


def import_indra_loader():
    global INDRA_LOADER
    global INDRA_API
    if INDRA_LOADER:
        return INDRA_LOADER
    if not importlib.util.find_spec("indra"):
        raise_indra_installation_error()  # type: ignore
    try:
        from indra import api  # type: ignore
        from indra.pytorch.loader import Loader  # type:ignore

        INDRA_API = api
        INDRA_LOADER = Loader
        return Loader
    except Exception as e:
        if not deeplake.constants.RETURN_DUMMY_DATA_FOR_DATALOADER:
            raise_indra_installation_error(e)
        INDRA_API = None
        INDRA_LOADER = None


class DeepLakeDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        _batch_size=None,
        _shuffle=None,
        _num_threads=None,
        _num_workers=None,
        _collate=None,
        _transform=None,
        _distributed=None,
        _prefetch_factor=None,
        _tensors=None,
        _drop_last=False,
        _mode=None,
        _return_index=None,
        _primary_tensor_name=None,
        _buffer_size=None,
        _orig_dataset=None,
        _decode_method=None,
        _persistent_workers=None,
        _dataloader=None,
        _world_size=1,
        _ignore_errors=False,
        _verbose=False,
        _offset=None,
        **kwargs,
    ):
        import_indra_loader()
        self.dataset = dataset
        self._orig_dataset = _orig_dataset or dataset
        self._batch_size = _batch_size
        self._shuffle = _shuffle
        self._num_threads = _num_threads
        self._num_workers = _num_workers
        self._collate = _collate
        self._transform = _transform
        self._distributed = _distributed
        self._prefetch_factor = _prefetch_factor
        self._tensors = _tensors
        self._drop_last = _drop_last
        self._mode = _mode
        self._return_index = _return_index
        self._primary_tensor_name = _primary_tensor_name or find_primary_tensor(dataset)
        self._buffer_size = _buffer_size
        self._decode_method = _decode_method
        self._persistent_workers = _persistent_workers
        self._dataloader = _dataloader
        self._world_size = _world_size
        self._ignore_errors = _ignore_errors
        self._verbose = _verbose
        self._offset = _offset
        for k, v in kwargs.items():
            setattr(self, k, v)

        # torch.utils.data.DataLoader attributes
        self.__initialized = True
        self._IterableDataset_len_called = None
        self._iterator = None
        self._worker_init_fn = None
        self.__multiprocessing_context = None

    @property
    def batch_size(self):
        return self._batch_size or 1

    @property
    def drop_last(self):
        return self._drop_last

    @property
    def num_workers(self):
        return self._num_workers or 0

    @property
    def prefetch_factor(self):
        return self._prefetch_factor or 0

    @property
    def pin_memory(self):
        return False

    @property
    def pin_memory_device(self):
        return ""

    @property
    def timeout(self):
        return 0

    @property
    def worker_init_fn(self):
        return self._worker_init_fn

    @worker_init_fn.setter
    def worker_init_fn(self, fn):
        self._worker_init_fn = fn
        if self._dataloader is not None:
            self._dataloader.worker_init_fn = fn

    @property  # type: ignore
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            (
                                "multiprocessing_context option "
                                "should specify a valid start method in {!r}, but got "
                                "multiprocessing_context={!r}"
                            ).format(valid_start_methods, multiprocessing_context)
                        )
                    multiprocessing_context = multiprocessing.get_context(
                        multiprocessing_context
                    )

                if not isinstance(
                    multiprocessing_context, multiprocessing.context.BaseContext
                ):
                    raise TypeError(
                        (
                            "multiprocessing_context option should be a valid context "
                            "object or a string specifying the start method, but got "
                            "multiprocessing_context={}"
                        ).format(multiprocessing_context)
                    )
            else:
                raise ValueError(
                    (
                        "multiprocessing_context can only be used with "
                        "multi-process loading (num_workers > 0), but got "
                        "num_workers={}"
                    ).format(self.num_workers)
                )

        self.__multiprocessing_context = multiprocessing_context

    @property
    def _dataset_kind(self):
        return 1

    @property
    def sampler(self):
        return (
            DistributedSampler(self._orig_dataset)
            if self._distributed
            else _InfiniteConstantSampler()
        )

    @property
    def batch_sampler(self):
        return (
            BatchSampler(self.sampler, self.batch_size, self.drop_last)
            if BatchSampler is not None
            else None
        )

    @property
    def generator(self):
        return None

    @property
    def persistent_workers(self):
        return self._persistent_workers or False

    @property
    def _auto_collation(self):
        return False

    @property
    def _index_sampler(self):
        return self.sampler

    @property
    def summary(self):
        if not self._ignore_errors or self._dataloader is None:
            return
        else:
            self._dataloader.summary

    @property
    def collate_fn(self):
        return get_collate_fn(self._collate, self._mode)

    def __len__(self):
        len_ds = (
            len(self._orig_dataset[self._tensors])
            if self._tensors is not None
            else len(self._orig_dataset)
        )
        round_fn = math.floor if self._drop_last else math.ceil
        return round_fn(len_ds / ((self.batch_size) * self._world_size))

    def batch(self, batch_size: int, drop_last: bool = False):
        """Returns a batched :class:`DeepLakeDataLoader` object.

        Args:
            batch_size (int): Number of samples in each batch.
            drop_last (bool): If True, the last batch will be dropped if its size is less than batch_size. Defaults to False.

        Returns:
            DeepLakeDataLoader: A :class:`DeepLakeDataLoader` object.

        Raises:
            ValueError: If .batch() has already been called.
        """
        if self._batch_size is not None:
            raise ValueError("batch size is already set")

        all_vars = self.__dict__.copy()
        all_vars["_batch_size"] = batch_size
        all_vars["_drop_last"] = drop_last
        return self.__class__(**all_vars)

    def offset(self, off: int = 0):
        """Returns a shifted :class:`DeepLakeDataLoader` object.

        Args:
            off (int): index that the dataloadee will start to iterate.

        Returns:
            DeepLakeDataLoader: A :class:`DeepLakeDataLoader` object.

        Raises:
            ValueError: If .offset() has already been called.
        """
        if self._offset is not None:
            raise ValueError("offset is already set")

        all_vars = self.__dict__.copy()
        all_vars["_offset"] = off
        return self.__class__(**all_vars)

    def shuffle(self, shuffle: bool = True, buffer_size: int = 2048):
        """Returns a shuffled :class:`DeepLakeDataLoader` object.

        Args:
            shuffle(bool): shows wheter we need to shuffle elements or not. Defaults to True.
            buffer_size (int): The size of the buffer used to shuffle the data in MBs. Defaults to 2048 MB. Increasing the buffer_size will increase the extent of shuffling.

        Returns:
            DeepLakeDataLoader: A :class:`DeepLakeDataLoader` object.

        Raises:
            ValueError: If .shuffle() has already been called.
            ValueError: If dataset is view and shuffle is True
        """
        if self._shuffle is not None:
            raise ValueError("shuffle is already set")
        all_vars = self.__dict__.copy()
        all_vars["_shuffle"] = shuffle
        all_vars["_buffer_size"] = buffer_size
        if shuffle:
            schedule = create_fetching_schedule(
                self._orig_dataset, self._primary_tensor_name
            )
            if schedule is not None:
                ds = self._orig_dataset  # type: ignore
                all_vars["_orig_dataset"] = ds[schedule]
        all_vars["_dataloader"] = None
        return self.__class__(**all_vars)

    def transform(
        self,
        transform: Union[Callable, Dict[str, Optional[Callable]]],
        **kwargs: Dict,
    ):
        """Returns a transformed :class:`DeepLakeDataLoader` object.


        Args:
            transform (Callable or Dict[Callable]): A function or dictionary of functions to apply to the data.
            kwargs: Additional arguments to be passed to `transform`. Only applicable if `transform` is a callable. Ignored if `transform` is a dictionary.

        Returns:
            DeepLakeDataLoader: A :class:`DeepLakeDataLoader` object.

        Raises:
            ValueError: If .transform() has already been called.
        """
        if self._transform is not None:
            raise ValueError("transform is already set")
        all_vars = self.__dict__.copy()
        if isinstance(transform, dict):
            tensors = [k for k in transform.keys() if k != "index"]
            tensors = map_tensor_keys(self._orig_dataset, tensors)
            if self._tensors:
                raise ValueError(
                    f"Tensors have already been specified in the .{self._mode} method."
                )
            all_vars["_tensors"] = map_tensor_keys(self._orig_dataset, tensors)
            transform = PytorchTransformFunction(transform_dict=transform)
        else:
            if kwargs:
                transform = partial(transform, **kwargs)
            transform = PytorchTransformFunction(composite_transform=transform)
        all_vars["_transform"] = transform
        all_vars["_dataloader"] = None
        return self.__class__(**all_vars)

    def query(self, query_string: str):
        """Returns a sliced :class:`DeepLakeDataLoader` object with given query results.
        It allows to run SQL like queries on dataset and extract results. See supported keywords and the Tensor Query Language documentation
        :ref:`here <tql>`.

        Args:
            query_string (str): An SQL string adjusted with new functionalities to run on the dataset object

        Returns:
            DeepLakeDataLoader: A :class:`DeepLakeDataLoader` object.

        Examples:
            >>> import deeplake
            >>> ds = deeplake.load('hub://activeloop/fashion-mnist-train')
            >>> query_ds_train = ds_train.dataloader().query("select * where labels != 5")

            >>> import deeplake
            >>> ds_train = deeplake.load('hub://activeloop/coco-train')
            >>> query_ds_train = ds_train.dataloader().query("(select * where contains(categories, 'car') limit 1000) union (select * where contains(categories, 'motorcycle') limit 1000)")
        """
        all_vars = self.__dict__.copy()
        all_vars["_orig_dataset"] = query(self._orig_dataset, query_string)
        all_vars["_dataloader"] = None
        return self.__class__(**all_vars)

    def sample_by(
        self,
        weights: Union[str, list, tuple, np.ndarray],
        replace: Optional[bool] = True,
        size: Optional[int] = None,
    ):
        """Returns a sliced :class:`DeepLakeDataLoader` with given weighted sampler applied

        Args:
            weights: (Union[str, list, tuple, np.ndarray]): If it's string then tql will be run to calculate the weights based on the expression. list, tuple and ndarray will be treated as the list of the weights per sample
            replace: Optional[bool] If true the samples can be repeated in the result view.
                (default: ``True``).
            size: Optional[int] The length of the result view.
                (default: ``len(dataset)``)

        Returns:
            DeepLakeDataLoader: A :class:`DeepLakeDataLoader` object.

        Examples:

            Sample the dataloader with ``labels == 5`` twice more than ``labels == 6``

            >>> ds = deeplake.load('hub://activeloop/fashion-mnist-train')
            >>> sampled_ds = ds.dataloader().sample_by("max_weight(labels == 5: 10, labels == 6: 5)")

            Sample the dataloader treating `labels` tensor as weights.

            >>> ds = deeplake.load('hub://activeloop/fashion-mnist-train')
            >>> sampled_ds = ds.dataloader().sample_by("labels")

            Sample the dataloader with the given weights;

            >>> ds_train = deeplake.load('hub://activeloop/coco-train')
            >>> weights = list()
            >>> for i in range(0, len(ds_train)):
            ...     weights.append(i % 5)
            ...
            >>> sampled_ds = ds.dataloader().sample_by(weights, replace=False)

        """
        all_vars = self.__dict__.copy()
        all_vars["_orig_dataset"] = sample_by(
            self._orig_dataset, weights, replace=replace, size=size
        )
        all_vars["_dataloader"] = None
        return self.__class__(**all_vars)

    def close(self):
        """Shuts down the workers and releases the resources."""
        if hasattr(self, "_iterator") and self._iterator is not None:
            self._iterator.close()
        if hasattr(self, "_dataloader") and self._dataloader is not None:
            self._dataloader = None

    def pytorch(
        self,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        tensors: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        prefetch_factor: int = 2,
        distributed: bool = False,
        return_index: bool = True,
        decode_method: Optional[Dict[str, str]] = None,
        persistent_workers: bool = False,
    ):
        """Creates a PyTorch Dataloader on top of the ``DeepLakeDataLoader`` from the Deep Lake dataset. During iteration, the data from all tensors will be streamed on-the-fly from the storage location.
        Understanding the parameters below is critical for achieving fast streaming for your use-case

        Args:
            num_workers (int): Number of workers to use for transforming and processing the data. Defaults to 0.
            collate_fn (Callable, Optional): merges a list of samples to form a mini-batch of Tensor(s).
            tensors (List, Optional): List of tensors to load. If ``None``, all tensors are loaded. Defaults to ``None``.
                For datasets with many tensors, its extremely important to stream only the data that is needed for training the model, in order to avoid bottlenecks associated with streaming unused data.
                For example, if you have a dataset that has ``image``, ``label``, and ``metadata`` tensors, if ``tensors=["image", "label"]``, the Data Loader will only stream the ``image`` and ``label`` tensors.
            num_threads (int, Optional): Number of threads to use for fetching and decompressing the data. If ``None``, the number of threads is automatically determined. Defaults to ``None``.
            prefetch_factor (int): Number of batches to transform and collate in advance per worker. Defaults to 2.
            distributed (bool): Used for DDP training. Distributes different sections of the dataset to different ranks. Defaults to ``False``.
            return_index (bool): Used to idnetify where loader needs to retur sample index or not. Defaults to ``True``.
            persistent_workers (bool): If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed once. Defaults to ``False``.
            decode_method (Dict[str, str], Optional): A dictionary of decode methods for each tensor. Defaults to ``None``.


                - Supported decode methods are:

                    :'numpy': Default behaviour. Returns samples as numpy arrays.
                    :'tobytes': Returns raw bytes of the samples.
                    :'pil': Returns samples as PIL images. Especially useful when transformation use torchvision transforms, that
                            require PIL images as input. Only supported for tensors with ``sample_compression='jpeg'`` or ``'png'``.

        Returns:
            DeepLakeDataLoader: A :class:`DeepLakeDataLoader` object.

        Raises:
            ValueError: If .pytorch() or .tensorflow() or .numpy() has already been called.
        
        Examples:
            
            >>> import deeplake
            >>> from torchvision import transforms
            >>> ds_train = deeplake.load('hub://activeloop/fashion-mnist-train')
            >>> tform = transforms.Compose([
            ...     transforms.RandomRotation(20), # Image augmentation
            ...     transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
            ...     transforms.Normalize([0.5], [0.5]),
            ... ])
            ...
            >>> batch_size = 32
            >>> # create dataloader by chaining with transform function and batch size and returns batch of pytorch tensors
            >>> train_loader = ds_train.dataloader()\\
            ...     .transform({'images': tform, 'labels': None})\\
            ...     .batch(batch_size)\\
            ...     .shuffle()\\
            ...     .pytorch(decode_method={'images': 'pil'}) # return samples as PIL images for transforms
            ...
            >>> # iterate over dataloader
            >>> for i, sample in enumerate(train_loader):
            ...     pass
            ...
        """
        import torch

        mode = "pytorch"
        handle_mode(self._mode, mode)
        all_vars = self.__dict__.copy()
        all_vars["_num_workers"] = num_workers
        all_vars["_collate"] = collate_fn
        validate_tensors(tensors, self._orig_dataset, all_vars)
        all_vars["_decode_method"] = decode_method
        all_vars["_num_threads"] = num_threads
        all_vars["_prefetch_factor"] = prefetch_factor
        all_vars["_distributed"] = distributed
        all_vars["_return_index"] = return_index
        all_vars["_mode"] = mode
        all_vars["_persistent_workers"] = persistent_workers
        all_vars["_dataloader"] = None
        if distributed:
            all_vars["_world_size"] = torch.distributed.get_world_size()
        return self.__class__(**all_vars)

    def tensorflow(
        self,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        tensors: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        prefetch_factor: int = 2,
        return_index: bool = True,
        decode_method: Optional[Dict[str, str]] = None,
        persistent_workers: bool = False,
    ):
        """Returns a :class:`DeepLakeDataLoader` object.


        Args:
            num_workers (int): Number of workers to use for transforming and processing the data. Defaults to 0.
            collate_fn (Callable, Optional): merges a list of samples to form a mini-batch of Tensor(s).
            tensors (List, Optional): List of tensors to load. If ``None``, all tensors are loaded. Defaults to ``None``.
                For datasets with many tensors, its extremely important to stream only the data that is needed for training the model, in order to avoid bottlenecks associated with streaming unused data.
                For example, if you have a dataset that has ``image``, ``label``, and ``metadata`` tensors, if ``tensors=["image", "label"]``, the Data Loader will only stream the ``image`` and ``label`` tensors.
            num_threads (int, Optional): Number of threads to use for fetching and decompressing the data. If ``None``, the number of threads is automatically determined. Defaults to ``None``.
            prefetch_factor (int): Number of batches to transform and collate in advance per worker. Defaults to 2.
            return_index (bool): If ``True``, the returned dataloader will have a key "index" that contains the index of the sample(s) in the original dataset. Default value is True.
            persistent_workers (bool): If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed once. Defaults to ``False``.
            decode_method (Dict[str, str], Optional): The method for decoding the Deep Lake tensor data, the result of which is passed to the transform. Decoding occurs outside of the transform so that it can be performed in parallel and as rapidly as possible as per Deep Lake optimizations.

                - Supported decode methods are:
                    :'numpy': Default behaviour. Returns samples as numpy arrays, the same as ds.tensor[i].numpy()
                    :'tobytes': Returns raw bytes of the samples the same as ds.tensor[i].tobytes()
                    :'data': Returns a dictionary with keys,values depending on htype, the same as ds.tensor[i].data()
                    :'pil': Returns samples as PIL images. Especially useful when transformation use torchvision transforms, that
                            require PIL images as input. Only supported for tensors with ``sample_compression='jpeg'`` or ``'png'``.

        Returns:
            DeepLakeDataLoader: A :class:`DeepLakeDataLoader` object.

        Raises:
            ValueError: If .pytorch() or .tensorflow() or .numpy() has already been called.
        
        Examples:
            
            >>> import deeplake
            >>> from torchvision import transforms
            >>> ds_train = deeplake.load('hub://activeloop/fashion-mnist-train')
            >>> batch_size = 32
            >>> # create dataloader by chaining with transform function and batch size and returns batch of pytorch tensors
            >>> train_loader = ds_train.dataloader()\\
            ...     .batch(batch_size)\\
            ...     .shuffle()\\
            ...     .tensorflow() # return samples as PIL images for transforms
            ...
            >>> # iterate over dataloader
            >>> for i, sample in enumerate(train_loader):
            ...     pass
            ...
        """
        import tensorflow as tf

        mode = "tensorflow"
        handle_mode(self._mode, mode)
        all_vars = self.__dict__.copy()
        all_vars["_num_workers"] = num_workers
        all_vars["_collate"] = collate_fn
        validate_tensors(tensors, self._orig_dataset, all_vars)
        all_vars["_decode_method"] = decode_method
        all_vars["_num_threads"] = num_threads
        all_vars["_prefetch_factor"] = prefetch_factor
        all_vars["_return_index"] = return_index
        all_vars["_mode"] = mode
        all_vars["_persistent_workers"] = persistent_workers
        all_vars["_dataloader"] = None
        return self.__class__(**all_vars)

    def numpy(
        self,
        num_workers: int = 0,
        tensors: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        prefetch_factor: int = 2,
        decode_method: Optional[Dict[str, str]] = None,
        persistent_workers: bool = False,
    ):
        """Returns a :class:`DeepLakeDataLoader` object.

        Args:
            num_workers (int): Number of workers to use for transforming and processing the data. Defaults to 0.
            tensors (List[str], Optional): List of tensors to load. If None, all tensors are loaded. Defaults to None.
            num_threads (int, Optional): Number of threads to use for fetching and decompressing the data. If None, the number of threads is automatically determined. Defaults to None.
            prefetch_factor (int): Number of batches to transform and collate in advance per worker. Defaults to 2.
            persistent_workers (bool): If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed once. Defaults to ``False``.
            decode_method (Dict[str, str], Optional): A dictionary of decode methods for each tensor. Defaults to None.

                - Supported decode methods are:-

                    :'numpy': Default behaviour. Returns samples as numpy arrays.
                    :'tobytes': Returns raw bytes of the samples.
                    :'pil': Returns samples as PIL images. Especially useful when transformation use torchvision transforms, that require PIL images as input. Only supported for tensors with sample_compression='jpeg' or 'png'.

        Returns:
            DeepLakeDataLoader: A :class:`DeepLakeDataLoader` object.

        Raises:
            ValueError: If .pytorch() or .tensorflow() or .numpy() has already been called.
        """
        mode = "numpy"
        handle_mode(self._mode, mode)
        all_vars = self.__dict__.copy()
        all_vars["_num_workers"] = num_workers
        validate_tensors(tensors, self._orig_dataset, all_vars)
        all_vars["_decode_method"] = decode_method
        all_vars["_tensors"] = self._tensors or tensors
        all_vars["_num_threads"] = num_threads
        all_vars["_prefetch_factor"] = prefetch_factor
        all_vars["_mode"] = mode
        all_vars["_persistent_workers"] = persistent_workers
        all_vars["_dataloader"] = None
        return self.__class__(**all_vars)

    def _get_suboptimal_thread_count(self) -> Optional[int]:
        assert self._distributed

        if self._num_threads is None and self._mode == "pytorch":
            import torch

            num_devices = (
                torch.cuda.device_count() if torch.cuda.is_available() else None
            )

            num_suboptimal_threads = (
                int(INDRA_API.num_available_threads() / num_devices)  # type: ignore [name-defined]
                if INDRA_API is not None and num_devices is not None  # type: ignore [name-defined]
                else None
            )
            return num_suboptimal_threads
        return self._num_threads

    def __is_upcast_needed(self, dataset, tensors) -> bool:
        for tensor_name in tensors:
            tensor = dataset._get_tensor_from_root(tensor_name)
            dtype = tensor.dtype
            if dtype is None:
                return True

            if dtype.type in [np.uint16, np.uint32, np.uint64]:
                return True

            if hasattr(dtype, "type"):
                if isinstance(
                    dtype.type, (type(np.uint16), type(np.uint32), type(np.uint64))
                ):
                    return True
        return False

    def __create_dummy_dataloader(
        self,
        dataset,
        tensors: Optional[List[str]] = None,
        raw_tensors: Optional[List[str]] = None,
        pil_compressed_tensors: Optional[List[str]] = None,
    ) -> DummyDataloader:
        return DummyDataloader(
            deeplake_dataset=dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            collate_fn=self.collate_fn,
            transform_fn=self._transform,
            distributed=self._distributed,
            prefetch_factor=self._prefetch_factor,
            tensors=tensors,
            drop_last=self._drop_last,
            upcast=self._mode == "pytorch"
            and self.__is_upcast_needed(
                dataset, tensors
            ),  # upcast to handle unsupported dtypes,
            return_index=self._return_index,
            raw_tensors=raw_tensors,
            pil_compressed_tensors=pil_compressed_tensors,
            persistent_workers=self._persistent_workers,
        )

    def __get_indra_dataloader(
        self,
        dataset,
        indra_dataset,
        tensors: Optional[List[str]] = None,
        raw_tensors: Optional[List[str]] = None,
        pil_compressed_tensors: Optional[List[str]] = None,
        json_tensors: Optional[List[str]] = None,
        list_tensors: Optional[List[str]] = None,
        htype_dict: Optional[dict] = None,
        ndim_dict: Optional[dict] = None,
        tensor_info_dict: Optional[dict] = None,
    ):
        num_threads = (
            self._get_suboptimal_thread_count()
            if self._distributed
            else self._num_threads
        )
        seed = DeeplakeRandom().get_seed()
        if self._offset is not None and self._shuffle and seed is None:
            warnings.warn(
                "offset and shuffle parameters are set without a random seed. This means that the ordering of the samples are "
                "not equal after each initialization and iteration through the dataloader. If you intend to shuffle data while "
                "preserving the offset for resuming iteration at a predictable index and order, please set a random seed using deeplake.random()"
            )
        from indra.pytorch.tensorinfo import TensorsInfo, LoaderMetaInfo  # type:ignore

        info = TensorsInfo(
            raw_tensors=raw_tensors or [],
            htype_dict=htype_dict or {},
            ndim_dict=ndim_dict or {},
            tensor_info_dict=tensor_info_dict or {},
            pil_compressed_tensors=pil_compressed_tensors or [],
            json_tensors=json_tensors or [],
            list_tensors=list_tensors or [],
        )

        loader_meta = LoaderMetaInfo(
            context=self.multiprocessing_context,
            distributed=self._distributed,
            upcast=self._mode == "pytorch"
            and self.__is_upcast_needed(
                dataset, tensors
            ),  # upcast to handle unsupported dtypes,
            return_index=self._return_index,
            verbose=self._verbose,
            ignore_errors=self._ignore_errors,
            prefetch_factor=self._prefetch_factor,
            offset=self._offset,
            primary_tensor=self._primary_tensor_name,
            worker_init_fn=self.worker_init_fn,
        )

        return INDRA_LOADER(  # type: ignore [misc]
            indra_dataset,
            batch_size=self._batch_size,
            num_threads=num_threads,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            collate_fn=self.collate_fn,
            transform_fn=self._transform,
            tensors=tensors,
            drop_last=self._drop_last,
            buffer_size=self._buffer_size,
            persistent_workers=self._persistent_workers,
            loader_meta=loader_meta,
            info=info,
        )

    def _fill_sample_info_tensors(
        self,
        dataset,
        sample_info_tensors,
        json_tensors,
        list_tensors,
    ):
        for tensor_name in sample_info_tensors:
            tensor = dataset._get_tensor_from_root(tensor_name)
            if len(tensor) == 0:
                raise EmptyTensorError(
                    f" the dataset has an empty tensor {tensor_name}, pytorch dataloader can't be created."
                    f" Please either populate the tensor or pass tensors argument to .pytorch that excludes this"
                    f" tensor."
                )
            meta = tensor.meta
            if meta.htype == "json":
                json_tensors.append(tensor_name)
            elif meta.htype in ["list", "tag"]:
                list_tensors.append(tensor_name)

    def __iter__(self):
        if self._dataloader is None:
            dataset = self._orig_dataset
            tensors = self._tensors or map_tensor_keys(dataset, None)

            jpeg_png_compressed_tensors, json_tensors, list_tensors = check_tensors(
                dataset, tensors
            )
            (
                raw_tensors,
                pil_compressed_tensors,
                json_tensors,
                list_tensors,
                data_tensors,
            ) = validate_decode_method(
                self._decode_method,
                tensors,
                jpeg_png_compressed_tensors,
                json_tensors,
                list_tensors,
            )
            sample_info_tensors, tensor_info_tensors = find_additional_tensors_and_info(
                dataset, data_tensors
            )
            self._fill_sample_info_tensors(
                dataset, sample_info_tensors, json_tensors, list_tensors
            )
            tensors.extend(sample_info_tensors)
            htype_dict, ndim_dict, tensor_info_dict = get_htype_ndim_tensor_info_dicts(
                dataset, data_tensors, tensor_info_tensors
            )
            if deeplake.constants.RETURN_DUMMY_DATA_FOR_DATALOADER:
                self._dataloader = self.__create_dummy_dataloader(
                    dataset,
                    tensors=tensors,
                    raw_tensors=raw_tensors,
                    pil_compressed_tensors=pil_compressed_tensors,
                )
            else:
                if not hasattr(self, "_indra_dataset"):
                    indra_dataset = dataset_to_libdeeplake(dataset)
                else:
                    indra_dataset = self._indra_dataset

                self._dataloader = self.__get_indra_dataloader(
                    dataset,
                    indra_dataset,
                    tensors=tensors,
                    raw_tensors=raw_tensors,
                    pil_compressed_tensors=pil_compressed_tensors,
                    json_tensors=json_tensors,
                    list_tensors=list_tensors,
                    htype_dict=htype_dict,
                    ndim_dict=ndim_dict,
                    tensor_info_dict=tensor_info_dict,
                )

        dataset_read(self._orig_dataset)

        if self._iterator is not None:
            self._iterator = iter(self._dataloader)

        return self

    def __setattr__(self, attr, val):
        if (
            attr == "_iterator"
            and val is None
            and hasattr(self, "_iterator")
            and self._iterator is not None
        ):
            self._iterator.close()
        else:
            super().__setattr__(attr, val)

    def __next__(self):
        if self._dataloader is None:
            self.__iter__()
        if self._iterator is None:
            self._iterator = iter(self._dataloader)
        return next(self._iterator)

    def __del__(self):
        self.close()


def dataloader(
    dataset, ignore_errors: bool = False, verbose: bool = False
) -> DeepLakeDataLoader:
    """Returns a :class:`~deeplake.enterprise.dataloader.DeepLakeDataLoader` object which can be transformed to either pytorch dataloader or numpy.


    Args:
        dataset: :class:`~deeplake.core.dataset.Dataset` object on which dataloader needs to be built
        ignore_errors (bool): If ``True``, the data loader will ignore errors appeared during data iteration otherwise it will collect the statistics and report appeared errors. Default value is ``False``
        verbose (bool): If ``True``, the data loader will dump verbose logs of it's steps. Default value is ``False``


    Returns:
        DeepLakeDataLoader: A :class:`~deeplake.enterprise.dataloader.DeepLakeDataLoader` object.


    Examples:


        Creating a simple dataloader object which returns a batch of numpy arrays


        >>> import deeplake
        >>> from deeplake.enterprise.dataloader import dataloader
        >>>
        >>> ds_train = deeplake.load('hub://activeloop/fashion-mnist-train')
        >>> train_loader = dataloader(ds_train).numpy()
        >>> for i, data in enumerate(train_loader):
        ...     # custom logic on data
        ...     pass


        Creating dataloader with custom transformation and batch size

        >>> import torch
        >>> from torchvision import datasets, transforms, models
        ...
        >>> ds_train = deeplake.load('hub://activeloop/fashion-mnist-train')
        >>> tform = transforms.Compose([
        ...     transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
        ...     transforms.RandomRotation(20), # Image augmentation
        ...     transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
        ...     transforms.Normalize([0.5], [0.5]),
        ... ])
        ...
        ...
        >>> batch_size = 32
        >>> #create dataloader with chaining transform function and batch size which returns batch of pytorch tensors
        >>> train_loader = dataloader(ds_train)
        ...     .transform({'images': tform, 'labels': None})
        ...     .batch(batch_size)
        ...     .shuffle()
        ...     .pytorch()
        ...
        >>> #loop over the elements
        >>> for i, data in enumerate(train_loader):
        ...     # custom logic on data
        ...     pass

        Creating dataloader and chaining with query

        >>> ds = deeplake.load('hub://activeloop/coco-train')
        >>> dl = dataloader(ds_train)
        ...     .query("(select * where contains(categories, 'car') limit 1000) union (select * where contains(categories, 'motorcycle') limit 1000)")
        ...     .pytorch()
        ...
        >>> #loop over the elements
        >>> for i, data in enumerate(train_loader):
        ...     # custom logic on data
        ...     pass
    """
    verify_base_storage(dataset)
    return DeepLakeDataLoader(dataset, _ignore_errors=ignore_errors, _verbose=verbose)


def validate_tensors(tensors, dataset, all_vars):
    existing_tensors = all_vars["_tensors"]
    if tensors:
        if "index" in tensors:
            raise ValueError(
                "index is not a tensor, to get index, pass return_index=True"
            )
        tensors = map_tensor_keys(dataset, tensors)
        if existing_tensors:
            raise ValueError(
                "Tensors have already been specified by passing a dictionary to .transform() method"
            )
    all_vars["_tensors"] = existing_tensors or tensors
