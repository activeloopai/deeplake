from typing import Callable, Dict, List, Optional, Union
from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake  # type: ignore
from deeplake.util.scheduling import create_fetching_schedule, find_primary_tensor
from deeplake.enterprise.util import (
    raise_indra_installation_error,
    verify_base_storage,
)
from deeplake.hooks import dataset_read
from deeplake.enterprise.libdeeplake_query import query, sample_by
from deeplake.integrations.pytorch.common import (
    PytorchTransformFunction,
    check_tensors,
    get_collate_fn,
    validate_decode_method,
)
from deeplake.util.dataset import map_tensor_keys
from functools import partial
import importlib
from torch.utils.data import DataLoader
import torch
import numpy as np

import math


# Load lazy to avoid cycylic import.
INDRA_LOADER = None


def indra_available() -> bool:
    try:
        import_indra_loader()
        return True
    except ImportError:
        return False


def import_indra_loader():
    global INDRA_LOADER
    if INDRA_LOADER:
        return INDRA_LOADER
    if not importlib.util.find_spec("indra"):
        raise_indra_installation_error()  # type: ignore
    try:
        from indra import api  # type: ignore
        from indra.pytorch.loader import Loader  # type:ignore

        INDRA_LOADER = Loader
        return Loader
    except Exception as e:
        raise_indra_installation_error(e)


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

    def __len__(self):
        round_fn = math.floor if self._drop_last else math.ceil
        return round_fn(
            len(self._orig_dataset) / ((self._batch_size or 1) * self._world_size)
        )

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
        if shuffle and isinstance(
            self.dataset.index.values[0].value, tuple  # type: ignore[attr-defined]
        ):
            raise ValueError("Can't shuffle dataset view")
        if self._shuffle is not None:
            raise ValueError("shuffle is already set")
        all_vars = self.__dict__.copy()
        all_vars["_shuffle"] = shuffle
        all_vars["_buffer_size"] = buffer_size
        if shuffle:
            schedule = create_fetching_schedule(self.dataset, self._primary_tensor_name)
            if schedule is not None:
                ds = self.dataset.no_view_dataset  # type: ignore
                all_vars["dataset"] = ds[schedule]
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
            tensors = map_tensor_keys(self.dataset, tensors)
            if self._tensors:
                raise ValueError(
                    f"Tensors have already been specified in the .{self._mode} method."
                )
            all_vars["_tensors"] = map_tensor_keys(self.dataset, tensors)
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
        all_vars["dataset"] = query(self.dataset, query_string)
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
        all_vars["dataset"] = sample_by(
            self.dataset, weights, replace=replace, size=size
        )
        all_vars["_dataloader"] = None
        return self.__class__(**all_vars)

    def close(self):
        """Shuts down the workers and releases the resources."""
        if self._dataloader is not None:
            self._dataloader.close()
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
        """Returns a :class:`DeepLakeDataLoader` object.


        Args:
            num_workers (int): Number of workers to use for transforming and processing the data. Defaults to 0.
            collate_fn (Callable, Optional): merges a list of samples to form a mini-batch of Tensor(s).
            tensors (List[str], Optional): List of tensors to load. If None, all tensors are loaded. Defaults to ``None``.
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
            ValueError: If .pytorch() or .numpy() has already been called.
        
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
        if self._mode is not None:
            if self._mode == "numpy":
                raise ValueError("Can't call .pytorch after .numpy()")
            raise ValueError("already called .pytorch()")
        all_vars = self.__dict__.copy()
        all_vars["_num_workers"] = num_workers
        all_vars["_collate"] = collate_fn
        validate_tensors(tensors, self.dataset, all_vars)
        all_vars["_decode_method"] = decode_method
        all_vars["_num_threads"] = num_threads
        all_vars["_prefetch_factor"] = prefetch_factor
        all_vars["_distributed"] = distributed
        all_vars["_return_index"] = return_index
        all_vars["_mode"] = "pytorch"
        all_vars["_persistent_workers"] = persistent_workers
        all_vars["_dataloader"] = None
        if distributed:
            all_vars["_world_size"] = torch.distributed.get_world_size()
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
            ValueError: If .pytorch() or .numpy() has already been called.
        """
        if self._mode is not None:
            if self._mode == "pytorch":
                raise ValueError("Can't call .numpy after .pytorch()")
            raise ValueError("already called .numpy()")
        all_vars = self.__dict__.copy()
        all_vars["_num_workers"] = num_workers
        validate_tensors(tensors, self.dataset, all_vars)
        all_vars["_decode_method"] = decode_method
        all_vars["_tensors"] = self._tensors or tensors
        all_vars["_num_threads"] = num_threads
        all_vars["_prefetch_factor"] = prefetch_factor
        all_vars["_mode"] = "numpy"
        all_vars["_persistent_workers"] = persistent_workers
        all_vars["_dataloader"] = None
        return self.__class__(**all_vars)

    def __iter__(self):
        if self._dataloader is None:
            collate_fn = get_collate_fn(self._collate, self._mode)
            upcast = self._mode == "pytorch"  # upcast to handle unsupported dtypes

            primary_tensor_name = self._primary_tensor_name
            buffer_size = self._buffer_size

            tensors = self._tensors or map_tensor_keys(self._orig_dataset, None)
            dataset = dataset_to_libdeeplake(self._orig_dataset)

            jpeg_png_compressed_tensors = check_tensors(self._orig_dataset, tensors)
            raw_tensors, compressed_tensors = validate_decode_method(
                self._decode_method, tensors, jpeg_png_compressed_tensors
            )
            raw_tensors.extend(compressed_tensors)
            self._dataloader = INDRA_LOADER(
                dataset,
                batch_size=self._batch_size,
                num_threads=self._num_threads,
                shuffle=self._shuffle,
                num_workers=self._num_workers,
                collate_fn=collate_fn,
                transform_fn=self._transform,
                distributed=self._distributed,
                prefetch_factor=self._prefetch_factor,
                tensors=tensors,
                drop_last=self._drop_last,
                upcast=upcast,
                return_index=self._return_index,
                primary_tensor=primary_tensor_name,
                buffer_size=buffer_size,
                raw_tensors=raw_tensors,
                compressed_tensors=compressed_tensors,
                persistent_workers=self._persistent_workers,
            )
        dataset_read(self._orig_dataset)
        return iter(self._dataloader)


def dataloader(dataset) -> DeepLakeDataLoader:
    """Returns a :class:`~deeplake.enterprise.dataloader.DeepLakeDataLoader` object which can be transformed to either pytorch dataloader or numpy.


    Args:
        dataset: :class:`~deeplake.core.dataset.Dataset` object on which dataloader needs to be built

    Returns:
        DeepLakeDataLoader: A :class:`~deeplake.enterprise.dataloader.DeepLakeDataLoader` object.


    Examples:


        Creating a simple dataloader object which returns a batch of numpy arrays


        >>> import deeplake
        >>> from deeplake.enterprise import dataloader
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

        Creating dataloader and chaning with query

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
    return DeepLakeDataLoader(dataset)


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
