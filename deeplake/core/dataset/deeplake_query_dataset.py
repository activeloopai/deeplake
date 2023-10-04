import posixpath
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from functools import partial
from deeplake.constants import SHOW_ITERATION_WARNING

from time import time
import math

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
from deeplake.util.scheduling import calculate_absolute_lengths
from deeplake.core.dataset import Dataset

import warnings

from deeplake.core.dataset.deeplake_query_tensor import DeepLakeQueryTensor


class DeepLakeQueryDataset(Dataset):
    def __init__(
        self,
        deeplake_ds,
        indra_ds,
        group_index=None,
        enabled_tensors=None,
        index: Optional[Index] = None,
    ):
        d: Dict[str, Any] = {}
        d["deeplake_ds"] = deeplake_ds
        d["indra_ds"] = indra_ds
        d["group_index"] = group_index or deeplake_ds.group_index
        d["enabled_tensors"] = enabled_tensors or deeplake_ds.enabled_tensors
        d["_index"] = index or deeplake_ds.index
        self.__dict__.update(d)

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
                deeplake_tensor = None
                try:
                    deeplake_tensor = self.deeplake_ds.__getattr__(fullpath)
                except:
                    pass
                indra_tensor = tensor
                return DeepLakeQueryTensor(
                    deeplake_tensor, indra_tensor, index=self.index
                )

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
        """
        # noqa: DAR101

        Raises:
            Exception: OSS dataloader is not supported on query dataset.
        """
        raise Exception(
            "OSS dataloader is not supported for non-linear views. Use `view.dataloader().pytorch()` instead."
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
            if self.deeplake_ds._has_group_in_root(fullpath):
                ret = DeepLakeQueryDataset(
                    deeplake_ds=self.deeplake_ds,
                    indra_ds=self.indra_ds,
                    index=self.index,
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
                    index=self.index,
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
                    deeplake_ds=self.deeplake_ds,
                    indra_ds=self.indra_ds[item],
                    index=self.index[item],
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
            try:
                return getattr(self.deeplake_ds, key)
            except AttributeError:
                raise AttributeError(
                    f"'{self.__class__}' object has no attribute '{key}'"
                ) from ke

    def __len__(self):
        return len(self.indra_ds)

    @deeplake_reporter.record_call
    def dataloader(self, ignore_errors: bool = False, verbose: bool = False):
        """Returns a :class:`~deeplake.enterprise.DeepLakeDataLoader` object. To use this, install deeplake with ``pip install deeplake[enterprise]``.

        Args:
            ignore_errors (bool): If ``True``, the data loader will ignore errors appeared during data iteration otherwise it will collect the statistics and report appeared errors. Default value is ``False``
            verbose (bool): If ``True``, the data loader will dump verbose logs of it's steps. Default value is ``False``

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

        dataloader = DeepLakeDataLoader(
            self,
            _indra_dataset=self.indra_ds,
            _ignore_errors=ignore_errors,
            _verbose=verbose,
        )
        return dataloader

    @property
    def no_view_dataset(self):
        return self

    @property
    def index(self):
        return self._index

    def _tensors(
        self, include_hidden: bool = True, include_disabled=True
    ) -> Dict[str, Tensor]:
        """All tensors belonging to this group, including those within sub groups. Always returns the sliced tensors."""
        original_tensors = self.deeplake_ds._tensors(include_hidden, include_disabled)
        indra_tensors = self.indra_ds.tensors
        indra_keys = set(t.name for t in indra_tensors)
        original_tensors = {
            k: v for k, v in original_tensors.items() if k in indra_keys or v.hidden
        }
        original_keys = set(original_tensors.keys())
        for t in indra_tensors:
            if t.name in original_keys:
                original_tensors[t.name] = DeepLakeQueryTensor(
                    original_tensors[t.name], t, index=self.index
                )
            else:
                original_tensors[t.name] = DeepLakeQueryTensor(
                    None, t, index=self.index
                )
        return original_tensors

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

    def __del__(self):
        """Leaving the implementation empty as at the moement indra dataset deletaion is taken care of in other place"""
        pass

    def random_split(self, lengths: Sequence[Union[int, float]]):
        if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
            lengths = calculate_absolute_lengths(lengths, len(self))

        vs = self.indra_ds.random_split(lengths)
        return [DeepLakeQueryDataset(self.deeplake_ds, v) for v in vs]
