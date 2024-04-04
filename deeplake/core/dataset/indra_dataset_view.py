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
from deeplake.core.meta.dataset_meta import DatasetMeta
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

from deeplake.core.dataset.indra_tensor_view import IndraTensorView


class IndraDatasetView(Dataset):
    def __init__(
        self,
        indra_ds,
        group_index="",
        enabled_tensors=None,
    ):
        d: Dict[str, Any] = {}
        d["indra_ds"] = indra_ds
        d["group_index"] = ""
        d["enabled_tensors"] = None
        d["verbose"] = False
        self.__dict__.update(d)
        self._view_base = None
        self._view_entry = None
        self._read_only = True
        self._locked_out = False
        self._query_string = None
        try:
            from deeplake.core.storage.indra import IndraProvider

            self.storage = IndraProvider(indra_ds.storage)
            self._read_only = self.storage.read_only
            self._token = self.storage.token
        except:
            pass

    @property
    def meta(self):
        return DatasetMeta()

    @property
    def path(self):
        try:
            return self.storage.original_path
        except:
            return ""

    @property
    def version_state(self) -> Dict:
        try:
            state = self.indra_ds.version_state
            for k, v in state["full_tensors"].items():
                state["full_tensors"][k] = IndraTensorView(v)
            return state
        except:
            return dict()

    @property
    def branches(self):
        return self.indra_ds.branches

    @property
    def commits(self) -> List[Dict]:
        return self.indra_ds.commits

    @property
    def commit_id(self) -> str:
        return self.indra_ds.commit_id

    @property
    def libdeeplake_dataset(self):
        return self.indra_ds

    def merge(self, *args, **kwargs):
        raise InvalidOperationError(
            "merge", "merge method cannot be called on a Dataset view."
        )

    def checkout(self, address: str, create: bool = False):
        if create:
            raise InvalidOperationError(
                "checkout", "Cannot create new branch on Dataset View."
            )
        self.indra_ds.checkout(address)

    def flush(self):
        pass

    def _get_tensor_from_root(self, fullpath):
        tensors = self.indra_ds.tensors
        for tensor in tensors:
            if tensor.name == fullpath:
                indra_tensor = tensor
                return IndraTensorView(indra_tensor)

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
            elif "/" in item:
                splt = posixpath.split(item)
                return self[splt[0]][splt[1]]
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
                return IndraDatasetView(
                    indra_ds=self.indra_ds,
                    enabled_tensors=enabled_tensors,
                )
            elif isinstance(item, tuple) and len(item) and isinstance(item[0], str):
                ret = self
                for x in item:
                    ret = self[x]
                return ret
            else:
                return IndraDatasetView(
                    indra_ds=self.indra_ds[item],
                )
        else:
            raise InvalidKeyTypeError(item)
        raise AttributeError("Dataset has no attribute - {item}")

    def __getattr__(self, key):
        try:
            ret = self.__getitem__(key)
        except AttributeError:
            ret = getattr(self.indra_ds, key)
        ret._view_entry = self._view_entry
        return ret

    def __len__(self):
        return len(self.indra_ds)

    @deeplake_reporter.record_call
    def dataloader(self, ignore_errors: bool = False, verbose: bool = False):
        """Returns a :class:`~deeplake.enterprise.dataloader.DeepLakeDataLoader` object.

        Args:
            ignore_errors (bool): If ``True``, the data loader will ignore errors appeared during data iteration otherwise it will collect the statistics and report appeared errors. Default value is ``False``
            verbose (bool): If ``True``, the data loader will dump verbose logs of it's steps. Default value is ``False``

        Returns:
            ~deeplake.enterprise.dataloader.DeepLakeDataLoader: A :class:`deeplake.enterprise.dataloader.DeepLakeDataLoader` object.
        
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
        from deeplake.enterprise.dataloader import DeepLakeDataLoader

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
    def base_storage(self):
        return self.storage

    @property
    def index(self):
        try:
            return Index(self.indra_ds.indexes)
        except:
            return Index(slice(0, len(self)))

    @property
    def sample_indices(self):
        for t in self.tensors.values():
            try:
                return t.indra_tensor.indexes
            except RuntimeError:
                pass
        return range(len(self))

    def _all_tensors_filtered(
        self, include_hidden: bool = True, include_disabled=True
    ) -> List[str]:
        indra_tensors = self.indra_ds.tensors
        return list(t.name for t in indra_tensors)

    def _tensors(
        self, include_hidden: bool = True, include_disabled=True
    ) -> Dict[str, IndraTensorView]:
        """All tensors belonging to this group, including those within sub groups. Always returns the sliced tensors."""
        indra_tensors = self.indra_ds.tensors
        ret = {}
        for t in indra_tensors:
            ret[t.name] = IndraTensorView(t)
        return ret

    def __str__(self):
        path_str = ""
        if self.path:
            path_str = f"path='{self.path}', "

        mode_str = ""
        if self.read_only:
            mode_str = f"read_only=True, "

        index_str = f"index={self.index}, "
        if self.index.is_trivial():
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
        return [IndraDatasetView(indra_ds=v) for v in vs]
