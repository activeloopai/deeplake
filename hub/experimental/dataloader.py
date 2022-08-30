from typing import Callable, List, Optional
from hub.experimental.convert_to_hub3 import dataset_to_hub3  # type: ignore
from hub.experimental.util import raise_indra_installation_error  # type: ignore
from hub.experimental.util import collate_fn as default_collate  # type: ignore
from hub.experimental.hub3_query import query
from hub.util.bugout_reporter import hub_reporter

try:
    from indra import Loader  # type: ignore

    INDRA_INSTALLED = True
except ImportError:
    INDRA_INSTALLED = False


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
    ):
        raise_indra_installation_error(INDRA_INSTALLED)
        # verifies underlying storage
        dataset_to_hub3(dataset)
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

    def batch(self, batch_size: int, drop_last: bool = False):
        """Returns a batched DataLoader object.

        Args:
            batch_size (int): Number of samples in each batch.
            drop_last (bool): If True, the last batch will be dropped if its size is less than batch_size. Defaults to False.

        Returns:
            Dataloader: A Dataloader object.

        Raises:
            ValueError: If .batch() has already been called.
        """
        if self._batch_size is not None:
            raise ValueError("batch size is already set")

        all_vars = self.__dict__.copy()
        all_vars["_batch_size"] = batch_size
        all_vars["_drop_last"] = drop_last
        return self.__class__(**all_vars)

    def shuffle(self):
        """Returns a shuffled Dataloader object.

        Returns:
            Dataloader: A Dataloader object.

        Raises:
            ValueError: If .shuffle() has already been called.
        """
        if self._shuffle is not None:
            raise ValueError("shuffle is already set")
        all_vars = self.__dict__.copy()
        all_vars["_shuffle"] = True
        return self.__class__(**all_vars)

    def transform(self, transform_fn: Callable):
        """Returns a transformed Dataloader object.

        Args:
            transform_fn (Callable): A function that takes a sample as input and returns a transformed sample.

        Returns:
            Dataloader: A Dataloader object.

        Raises:
            ValueError: If .transform() has already been called.
        """
        if self._transform is not None:
            raise ValueError("transform is already set")
        all_vars = self.__dict__.copy()
        all_vars["_transform"] = transform_fn
        return self.__class__(**all_vars)

    def query(self, query_string: str):
        all_vars = self.__dict__.copy()
        all_vars["dataset"] = query(self.dataset, query_string)
        return self.__class__(**all_vars)

    def pytorch(
        self,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        tensors: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        prefetch_factor: int = 10,
        distributed: bool = False,
    ):
        """Returns a pytorch Dataloader object.

        Args:
            num_workers (int): Number of workers to use for transforming and processing the data. Defaults to 0.
            collate_fn (Callable, Optional): merges a list of samples to form a mini-batch of Tensor(s).
            tensors (List[str], Optional): List of tensors to load. If None, all tensors are loaded. Defaults to None.
            num_threads (int, Optional): Number of threads to use for fetching and decompressing the data. If None, the number of threads is automatically determined. Defaults to None.
            prefetch_factor (int): Number of batches to transform and collate in advance per worker. Defaults to 10.
            distributed (bool): Used for DDP training. Distributes different sections of the dataset to different ranks. Defaults to False.

        Returns:
            Dataloader: A Dataloader object.

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
        all_vars["_tensors"] = tensors
        all_vars["_num_threads"] = num_threads
        all_vars["_prefetch_factor"] = prefetch_factor
        all_vars["_distributed"] = distributed
        all_vars["_mode"] = "pytorch"
        return self.__class__(**all_vars)

    def numpy(
        self,
        num_workers: int = 0,
        tensors: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        prefetch_factor: int = 10,
    ):
        """Returns a numpy Dataloader object.

        Args:
            num_workers (int): Number of workers to use for transforming and processing the data. Defaults to 0.
            tensors (List[str], Optional): List of tensors to load. If None, all tensors are loaded. Defaults to None.
            num_threads (int, Optional): Number of threads to use for fetching and decompressing the data. If None, the number of threads is automatically determined. Defaults to None.
            prefetch_factor (int): Number of batches to transform and collate in advance per worker. Defaults to 10.

        Returns:
            Dataloader: A Dataloader object.

        Raises:
            ValueError: If .to_pytorch() or .to_numpy() has already been called.
        """
        if self._mode is not None:
            if self._mode == "pytorch":
                raise ValueError("Can't call .to_numpy after .to_pytorch()")
            raise ValueError("already called .to_numpy()")
        all_vars = self.__dict__.copy()
        all_vars["_num_workers"] = num_workers
        all_vars["_tensors"] = tensors
        all_vars["_num_threads"] = num_threads
        all_vars["_prefetch_factor"] = prefetch_factor
        all_vars["_mode"] = "numpy"
        return self.__class__(**all_vars)

    def __iter__(self):
        dataset = dataset_to_hub3(self.dataset)
        batch_size = self._batch_size or 1
        drop_last = self._drop_last or False

        shuffle = self._shuffle or False

        transform_fn = self._transform

        num_workers = self._num_workers or 0
        if self._collate is None and self._mode == "pytorch":
            collate_fn = default_collate
        else:
            collate_fn = self._collate
        tensors = self._tensors or []
        num_threads = self._num_threads
        prefetch_factor = self._prefetch_factor
        distributed = self._distributed or False
        upcast = (
            self._mode == "pytorch"
        )  # only upcast for pytorch, this handles unsupported dtypes
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
            )
        )


@hub_reporter.record_call
def dataloader(dataset) -> Hub3DataLoader:
    return Hub3DataLoader(dataset)
