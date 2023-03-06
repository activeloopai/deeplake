import posixpath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial
from deeplake.constants import SHOW_ITERATION_WARNING

from time import time

from deeplake.util.iteration_warning import (
    check_if_iteration,
)
from deeplake.api.info import load_info
from deeplake.client.log import logger
from deeplake.client.utils import get_user_name
from deeplake.constants import (
    SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE,
)
from deeplake.core.index import Index
from deeplake.core.tensor import Tensor
from deeplake.util.bugout_reporter import deeplake_reporter
from deeplake.util.exceptions import (
    InvalidKeyTypeError,
    InvalidOperationError,
    MemoryDatasetCanNotBePickledError,
    TensorDoesNotExistError,
)
from deeplake.core.dataset import Dataset

import warnings

from deeplake.core.dataset.deeplake_query_tensor import DeepLakeQueryTensor


class DeepLakeQueryDataset(Dataset):
    def __init__(self, deeplake_ds, indra_ds, group_index=None, enabled_tensors=None):
        self.deeplake_ds = deeplake_ds
        self.indra_ds = indra_ds
        self.group_index = group_index or deeplake_ds.group_index
        self.enabled_tensors = enabled_tensors or deeplake_ds.enabled_tensors
        self.set_deeplake_dataset_variables()

    def set_deeplake_dataset_variables(self):
        self.path = self.deeplake_ds.path
        if self.path.startswith("mem://"):
            raise MemoryDatasetCanNotBePickledError

        keys = [
            "base_storage",
            "_read_only",
            "public",
            "storage",
            "_token",
            "verbose",
            "version_state",
            "org_id",
            "ds_name",
            "_is_filtered_view",
            "_view_id",
            "_view_use_parent_commit",
            "_parent_dataset",
            "_pad_tensors",
            "_locking_enabled",
            "is_iteration",
            "_view_base",
            "link_creds",
            "_locked_out",
            "_indexing_history",
        ]

        for k in keys:
            setattr(self, k, getattr(self.deeplake_ds, k))

    @property
    def meta(self):
        return self.deeplake_ds.meta

    def merge(self, *args, **kwargs):
        raise InvalidOperationError(
            "merge", "merge method cannot be called on a Dataset view."
        )

    def checkout(self, address: str, create: bool = False):
        raise InvalidOperationError(
            "checkout", "checkout method cannot be called on a Dataset view."
        )

    def _get_tensor_from_root(self, fullpath):
        tensors = self.indra_ds.tensors
        for tensor in tensors:
            if tensor.name == fullpath:
                deeplake_tensor = self.deeplake_ds.__getattr__(fullpath)
                indra_tensor = tensor
                return DeepLakeQueryTensor(deeplake_tensor, indra_tensor)

    def pytorch(
        self,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        return_index: bool = True,
        transform: Optional[Callable] = None,
        num_workers: int = 0,
        num_threads: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        distributed=False,
        tensors: Optional[List[str]] = None,
        raw_tensors: Optional[List[str]] = None,
        compressed_tensors: Optional[List[str]] = None,
        prefetch_factor: int = 10,
        upcast: bool = True,
        primary_tensor: Optional[str] = None,
        buffer_size: int = 2048,
        persistent_workers: bool = False,
    ):
        """Returns a dataloader object referencing to C++ dataloader instance.

        Args:
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            shuffle (bool): set to ``True`` to have the data reshuffled at every epoch
                (default: ``False``).
            drop_last (bool): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: ``False``)
            return_index (bool): Showing wheter Loader needs to return the sample index during iteration.Defaults to True.
            transform (Callable, optional): Callable object which is needed to be applyed on each sample on batch. Defaults to None.
            num_workers (int): How many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main process.
                (default: ``0``)
            num_threads (int, optional): Number of threads that nedes to be spinned up during data loading. Defaults to None.
                if it is none then we are detecting the hardware concurency count to set.
                Note: We don't set any threading flags (eg. OMP_NUM_THREADS, MKL_NUM_THREADS, etc)
                to get more performant Loader consider of not setting those flags which can affect on 3rd party libraries worflow performance
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            distributed (bool): Flag that is showing whether Loader needes to work in DDP or not. Defaults to ``False``
            tensors (List[str], optinal): List of tensors thet are participating to in Loadeing process.
                Defaults to ``None`` which means that Loader will fetch samples for all of the tensors
            raw_tensors (List[str], optional): List of the tensors that needs to return raw data instead of decompression.
                Defaults to ``None`` if raw_tensors is None then all the tensors will send decompression data
                E.g raw_tensors['images'] then only the images tensor data will be sent as a row array
            compressed_tensors (List[str], optional): Subset of raw tensors, these will be decompressed by python workers.
            prefetch_factor (int): Number of samples loaded in advance by workers. Defaults to 10
            upcast (bool): Flag that is showing wheter we need to upcast object if dtype is not supported this is needed only for
                pytorch as it is not support all the dtypes. Defaults to True.
            primary_tensor (Optional[str]): Name of primary tensor.
            buffer_size (int): The size of the buffer used to shuffle the data in MBs. Defaults to 2048 MB. Increasing the buffer_size will increase the extent of shuffling.
            persistent_workers (bool): If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed once. Defaults to ``False``.

        Raises:
            Exception: OSS dataloader is not supported on query dataset.
        """
        raise Exception(
            "OSS dataloader is not supported for non-linear views. Use `view.dataloader().pytorch` instead."
        )

    def __getitem__(
        self,
        item: Union[
            str, int, slice, List[int], Tuple[Union[int, slice, Tuple[int]]], Index
        ],
        is_iteration: bool = False,
    ):
        if isinstance(item, str):
            fullpath = posixpath.join(self.group_index, item)
            enabled_tensors = self.enabled_tensors
            if enabled_tensors is None or fullpath in enabled_tensors:
                tensor = self._get_tensor_from_root(fullpath)
                if tensor is not None:
                    return tensor
            if self._has_group_in_root(fullpath):
                ret = DeepLakeQueryDataset(
                    deeplake_ds=self.deeplake_ds,
                    indra_ds=self.indra_ds,
                    group_index=posixpath.join(self.group_index, item),
                )
            elif "/" in item:
                splt = posixpath.split(item)
                ret = self[splt[0]][splt[1]]
            else:
                raise TensorDoesNotExistError(item)
        elif isinstance(item, (int, slice, list, tuple, Index, type(Ellipsis))):
            if (
                isinstance(item, list)
                and len(item)
                and (
                    isinstance(item[0], str)
                    or (
                        isinstance(item[0], (list, tuple))
                        and len(item[0])
                        and isinstance(item[0][0], str)
                    )
                )
            ):
                group_index = self.group_index
                enabled_tensors = [
                    posixpath.join(
                        group_index, (x if isinstance(x, str) else "/".join(x))
                    )
                    for x in item
                ]
                ret = DeepLakeQueryDataset(
                    deeplake_ds=self.deeplake_ds,
                    indra_ds=self.indra_ds,
                    enabled_tensors=enabled_tensors,
                )
            elif isinstance(item, tuple) and len(item) and isinstance(item[0], str):
                ret = self
                for x in item:
                    ret = self[x]
                return ret
            else:
                if not is_iteration and isinstance(item, int):
                    is_iteration = check_if_iteration(self._indexing_history, item)
                    if is_iteration and SHOW_ITERATION_WARNING:
                        warnings.warn(
                            "Indexing by integer in a for loop, like `for i in range(len(ds)): ... ds[i]` can be quite slow. Use `for i, sample in enumerate(ds)` instead."
                        )
                ret = DeepLakeQueryDataset(
                    deeplake_ds=self.deeplake_ds[item],
                    indra_ds=self.indra_ds[item],
                )
        else:
            raise InvalidKeyTypeError(item)

        if hasattr(self, "_view_entry"):
            ret._view_entry = self._view_entry
        return ret

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except TensorDoesNotExistError as ke:
            raise AttributeError(
                f"'{self.__class__}' object has no attribute '{key}'"
            ) from ke

    def __len__(self):
        return len(self.indra_ds)

    @deeplake_reporter.record_call
    def dataloader(self):
        """Returns a :class:`~deeplake.enterprise.DeepLakeDataLoader` object. To use this, install deeplake with ``pip install deeplake[enterprise]``.

        Returns:
            ~deeplake.enterprise.DeepLakeDataLoader: A :class:`deeplake.enterprise.DeepLakeDataLoader` object.
        
        Examples:

            Creating a simple dataloader object which returns a batch of numpy arrays

            >>> import deeplake
            >>> ds_train = deeplake.load('hub://activeloop/fashion-mnist-train')
            >>> train_loader = ds_train.dataloader().numpy()
            >>> for i, data in enumerate(train_loader):
            ...     # custom logic on data
            ...     pass


            Creating dataloader with custom transformation and batch size

            >>> import deeplake
            >>> import torch
            >>> from torchvision import datasets, transforms, models
            >>> 
            >>> ds_train = deeplake.load('hub://activeloop/fashion-mnist-train')
            >>> tform = transforms.Compose([
            ...     transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
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
            ...     .pytorch()
            ...
            >>> # loop over the elements
            >>> for i, data in enumerate(train_loader):
            ...     # custom logic on data
            ...     pass

            Creating dataloader and chaining with query

            >>> ds = deeplake.load('hub://activeloop/coco-train')
            >>> train_loader = ds_train.dataloader()\\
            ...     .query("(select * where contains(categories, 'car') limit 1000) union (select * where contains(categories, 'motorcycle') limit 1000)")\\
            ...     .pytorch()
            ...
            >>> # loop over the elements
            >>> for i, data in enumerate(train_loader):
            ...     # custom logic on data
            ...     pass

        **Restrictions**

        The new high performance C++ dataloader is part of our Growth and Enterprise Plan .

        - Users of our Community plan can create dataloaders on Activeloop datasets ("hub://activeloop/..." datasets).
        - To run queries on your own datasets, `upgrade your organization's plan <https://www.activeloop.ai/pricing/>`_.
        """
        from deeplake.enterprise import DeepLakeDataLoader

        dataloader = DeepLakeDataLoader(self, _indra_dataset=self.indra_ds)
        return dataloader

    @property
    def index(self):
        return self.deeplake_ds.index

    def _tensors(
        self, include_hidden: bool = True, include_disabled=True
    ) -> Dict[str, Tensor]:
        """All tensors belonging to this group, including those within sub groups. Always returns the sliced tensors."""
        version_state = self.version_state
        group_index = self.group_index
        all_tensors = self._all_tensors_filtered(include_hidden, include_disabled)
        return {t: self[posixpath.join(group_index, t)] for t in all_tensors}

    def __str__(self):
        path_str = ""
        if self.path:
            path_str = f"path='{self.path}', "

        mode_str = ""
        if self.read_only:
            mode_str = f"read_only=True, "

        index_str = f"index={self.deeplake_ds.index}, "
        if self.deeplake_ds.index.is_trivial():
            index_str = ""

        group_index_str = (
            f"group_index='{self.group_index}', " if self.group_index else ""
        )

        return f"Dataset({path_str}{mode_str}{index_str}{group_index_str}tensors={self._all_tensors_filtered(include_hidden=False, include_disabled=False)})"

    def copy(self, *args, **kwargs):
        raise NotImplementedError(
            "Copying or Deepcopying for views generated by nonlinear queries is not supported."
        )

    def _create_sample_shape_tensor(self, tensor: str):
        shape_tensor_name = f"_{tensor}_shape"
        shape_tensor = self.create_tensor(
            shape_tensor_name,
            dtype="int64",
            hidden=True,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            max_chunk_size=SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE,
        )

        indra_tensor = getattr(self.indra_ds, tensor)
        shape_tensor.extend(indra_tensor)
