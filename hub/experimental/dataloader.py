from typing import Callable, Dict, List, Optional, Sequence, Union
from hub.experimental.convert_to_hub3 import dataset_to_hub3
from hub.experimental.util import (
    create_fetching_schedule,
    find_primary_tensor,
    raise_indra_installation_error,
    verify_base_storage,
)
from hub.experimental.util import collate_fn as default_collate  # type: ignore
from hub.experimental.hub3_query import query
from hub.integrations.pytorch.common import PytorchTransformFunction
from hub.util.bugout_reporter import hub_reporter
from hub.util.dataset import map_tensor_keys
import importlib

INDRA_INSTALLED = bool(importlib.util.find_spec("indra"))

if INDRA_INSTALLED:
    try:
        from indra.pytorch.loader import Loader  # type:ignore

        INDRA_IMPORT_ERROR = None
    except ImportError as e:
        INDRA_IMPORT_ERROR = e


class Hub3DataLoader:
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
        _to_bytes=None,
    ):
        raise_indra_installation_error(INDRA_INSTALLED, INDRA_IMPORT_ERROR)
        self.dataset = dataset
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
        self._to_bytes = _to_bytes

    def batch(self, batch_size: int, drop_last: bool = False):
        """Returns a batched :class:`Hub3DataLoader` object.

        Args:
            batch_size (int): Number of samples in each batch.
            drop_last (bool): If True, the last batch will be dropped if its size is less than batch_size. Defaults to False.

        Returns:
            Hub3DataLoader: A :class:`Hub3DataLoader` object.

        Raises:
            ValueError: If .batch() has already been called.
        """
        if self._batch_size is not None:
            raise ValueError("batch size is already set")

        all_vars = self.__dict__.copy()
        all_vars["_batch_size"] = batch_size
        all_vars["_drop_last"] = drop_last
        return self.__class__(**all_vars)

    def shuffle(self, buffer_size: int = 2048):
        """Returns a shuffled :class:`Hub3DataLoader` object.

        Args:
            buffer_size (int): The size of the buffer used to shuffle the data in MBs. Defaults to 2048 MB. Increasing the buffer_size will increase the extent of shuffling.

        Returns:
            Hub3DataLoader: A :class:`Hub3DataLoader` object.

        Raises:
            ValueError: If .shuffle() has already been called.
        """
        if self._shuffle is not None:
            raise ValueError("shuffle is already set")
        all_vars = self.__dict__.copy()
        all_vars["_shuffle"] = True
        all_vars["_buffer_size"] = buffer_size
        schedule = create_fetching_schedule(self.dataset, self._primary_tensor_name)
        if schedule is not None:
            all_vars["dataset"] = self.dataset[schedule]
        return self.__class__(**all_vars)

    def transform(self, transform: Union[Callable, Dict[str, Optional[Callable]]]):
        """Returns a transformed :class:`Hub3DataLoader` object.


        Args:
            transform (Callable or Dict[Callable]): A function or dictionary of functions to apply to the data.

        Returns:
            Hub3DataLoader: A :class:`Hub3DataLoader` object.

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
            transform = PytorchTransformFunction(composite_transform=transform)
        all_vars["_transform"] = transform
        return self.__class__(**all_vars)

    def query(self, query_string: str):
        """Returns a sliced :class:`Hub3DataLoader` object with given query results.
        It allows to run SQL like queries on dataset and extract results. See supported keywords and the Tensor Query Language documentation
        :ref:`here <tql>`.

        Args:
            query_string (str): An SQL string adjusted with new functionalities to run on the dataset object

        Returns:
            Hub3DataLoader: A :class:`Hub3DataLoader` object.

        Examples:
            >>> import hub
            >>> from hub.experimental import dataloader
            >>> ds = hub.load('hub://activeloop/fashion-mnist-train')
            >>> query_ds_train = dataloader(ds_train).query("select * where labels != 5")

            >>> import hub
            >>> from hub.experimental import query
            >>> ds_train = hub.load('hub://activeloop/coco-train')
            >>> query_ds_train = dataloader(ds_train).query("(select * where contains(categories, 'car') limit 1000) union (select * where contains(categories, 'motorcycle') limit 1000)")
        """
        all_vars = self.__dict__.copy()
        all_vars["dataset"] = query(self.dataset, query_string)
        return self.__class__(**all_vars)

    @hub_reporter.record_call
    def pytorch(
        self,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        tensors: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        prefetch_factor: int = 10,
        distributed: bool = False,
        return_index: bool = True,
        tobytes: Union[bool, Sequence[str]] = False,
    ):
        """Returns a :class:`Hub3DataLoader` object.


        Args:
            num_workers (int): Number of workers to use for transforming and processing the data. Defaults to 0.
            collate_fn (Callable, Optional): merges a list of samples to form a mini-batch of Tensor(s).
            tensors (List[str], Optional): List of tensors to load. If None, all tensors are loaded. Defaults to None.
            num_threads (int, Optional): Number of threads to use for fetching and decompressing the data. If None, the number of threads is automatically determined. Defaults to None.
            prefetch_factor (int): Number of batches to transform and collate in advance per worker. Defaults to 10.
            distributed (bool): Used for DDP training. Distributes different sections of the dataset to different ranks. Defaults to False.
            return_index (bool): Used to idnetify where loader needs to retur sample index or not. Defaults to True.
            tobytes (bool, Sequence[str]): If ``True``, samples will not be decompressed and their raw bytes will be returned instead of numpy arrays. Can also be a list of tensors, in which case those tensors alone will not be decompressed.

        Returns:
            Hub3DataLoader: A :class:`Hub3DataLoader` object.

        Raises:
            ValueError: If .to_pytorch() or .to_numpy() has already been called.
        """
        if self._mode is not None:
            if self._mode == "numpy":
                raise ValueError("Can't call .to_pytorch after .to_numpy()")
            raise ValueError("already called .to_pytorch()")
        all_vars = self.__dict__.copy()
        all_vars["_num_workers"] = num_workers
        all_vars["_collate"] = collate_fn
        handle_tensors_and_tobytes(tensors, tobytes, self.dataset, all_vars)
        all_vars["_num_threads"] = num_threads
        all_vars["_prefetch_factor"] = prefetch_factor
        all_vars["_distributed"] = distributed
        all_vars["_return_index"] = return_index
        all_vars["_mode"] = "pytorch"
        return self.__class__(**all_vars)

    @hub_reporter.record_call
    def numpy(
        self,
        num_workers: int = 0,
        tensors: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        prefetch_factor: int = 10,
        tobytes: Union[bool, Sequence[str]] = False,
    ):
        """Returns a :class:`Hub3DataLoader` object.

        Args:
            num_workers (int): Number of workers to use for transforming and processing the data. Defaults to 0.
            tensors (List[str], Optional): List of tensors to load. If None, all tensors are loaded. Defaults to None.
            num_threads (int, Optional): Number of threads to use for fetching and decompressing the data. If None, the number of threads is automatically determined. Defaults to None.
            prefetch_factor (int): Number of batches to transform and collate in advance per worker. Defaults to 10.
            tobytes (bool, Sequence[str]): If ``True``, samples will not be decompressed and their raw bytes will be returned instead of numpy arrays. Can also be a list of tensors, in which case those tensors alone will not be decompressed.

        Returns:
            Hub3DataLoader: A :class:`Hub3DataLoader` object.

        Raises:
            ValueError: If .to_pytorch() or .to_numpy() has already been called.
        """
        if self._mode is not None:
            if self._mode == "pytorch":
                raise ValueError("Can't call .to_numpy after .to_pytorch()")
            raise ValueError("already called .to_numpy()")
        all_vars = self.__dict__.copy()
        all_vars["_num_workers"] = num_workers
        handle_tensors_and_tobytes(tensors, tobytes, self.dataset, all_vars)
        all_vars["_tensors"] = self._tensors or tensors
        all_vars["_num_threads"] = num_threads
        all_vars["_prefetch_factor"] = prefetch_factor
        all_vars["_mode"] = "numpy"
        return self.__class__(**all_vars)

    def __iter__(self):
        dataset = dataset_to_hub3(self.dataset)
        batch_size = self._batch_size or 1
        drop_last = self._drop_last or False
        return_index = self._return_index
        if return_index is None:
            return_index = True

        shuffle = self._shuffle or False

        transform_fn = self._transform

        num_workers = self._num_workers or 0
        if self._collate is None and self._mode == "pytorch":
            collate_fn = default_collate
        else:
            collate_fn = self._collate
        tensors = self._tensors or map_tensor_keys(self.dataset, None)
        num_threads = self._num_threads
        prefetch_factor = self._prefetch_factor
        distributed = self._distributed or False

        # only upcast for pytorch, this handles unsupported dtypes
        upcast = self._mode == "pytorch"

        primary_tensor_name = self._primary_tensor_name
        buffer_size = self._buffer_size
        if self._to_bytes is True:
            raw_tensors = tensors
        elif self._to_bytes is False:
            raw_tensors = []
        else:
            raw_tensors = self._to_bytes
        return iter(
            Loader(
                dataset,
                batch_size=batch_size,
                num_threads=num_threads,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                transform_fn=transform_fn,
                distributed=distributed,
                prefetch_factor=prefetch_factor,
                tensors=tensors,
                drop_last=drop_last,
                upcast=upcast,
                return_index=return_index,
                primary_tensor=primary_tensor_name,
                buffer_size=buffer_size,
                raw_tensors=raw_tensors,
            )
        )


def dataloader(dataset) -> Hub3DataLoader:
    """Returns a :class:`Hub3DataLoader` object which can be transformed to either pytorch dataloader or numpy.


    Args:
        dataset: hub.Dataset object on which dataloader needs to be built

    Returns:
        Hub3DataLoader: A :class:`Hub3DataLoader` object.


    Examples:


        Creating a simple dataloader object which returns a batch of numpy arrays


        >>> import hub
        >>> from hub.experimental import dataloader
        >>>
        >>> ds_train = hub.load('hub://activeloop/fashion-mnist-train')
        >>> train_loader = dataloader(ds_train).numpy()
        >>> for i, data in enumerate(train_loader):
        ...     # custom logic on data
        ...     pass


        Creating dataloader with custom transformation and batch size

        >>> import torch
        >>> from torchvision import datasets, transforms, models
        ...
        >>> ds_train = hub.load('hub://activeloop/fashion-mnist-train')
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
        ...     .pytorch()
        ...
        >>> #loop over the elements
        >>> for i, data in enumerate(train_loader):
        ...     # custom logic on data
        ...     pass

        Creating dataloader and chaning with query

        >>> ds = hub.load('hub://activeloop/coco-train')
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
    return Hub3DataLoader(dataset)


def handle_tensors_and_tobytes(tensors, tobytes, dataset, all_vars):
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
    if isinstance(tobytes, Sequence):
        tobytes = map_tensor_keys(dataset, tobytes)
        if tobytes and all_vars["_tensors"]:
            tensor_set = set(all_vars["_tensors"])
            for tensor in tobytes:
                if tensor not in tensor_set:
                    raise ValueError(f"tobytes tensor {tensor} not found in tensors.")
    all_vars["_tobytes"] = tobytes
