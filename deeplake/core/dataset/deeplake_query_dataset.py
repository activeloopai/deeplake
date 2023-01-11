import posixpath
import warnings
from typing import List, Tuple, Union, Dict, Optional, Callable

import deeplake.core.dataset as dataset
from deeplake.util.exceptions import (
    InvalidOperationError,
    TensorDoesNotExistError,
    InvalidKeyTypeError,
    MemoryDatasetCanNotBePickledError,
)
import deeplake
from deeplake.core.index import Index

try:
    from indra.pytorch.loader import Loader
    from indra.pytorch.common import collate_fn as default_collate

    _INDRA_INSTALLED = True
except ImportError:
    _INDRA_INSTALLED = False
from deeplake.core.dataset.deeplake_query_tensor import DeepLakeQueryTensor
from deeplake.util.iteration_warning import (
    check_if_iteration,
)

from deeplake.core.dataset.view_entry import ViewEntry
from deeplake.util.logging import log_visualizer_link


class DeepLakeQueryDataset(dataset.Dataset):
    def __init__(
        self, deeplake_ds, indra_ds, group_index=None, enabled_tensors=None, index=None
    ):
        self.deeplake_ds = deeplake_ds
        self.indra_ds = indra_ds
        self.group_index = group_index or deeplake_ds.group_index
        self.enabled_tensors = enabled_tensors or deeplake_ds.enabled_tensors
        self.index = index or deeplake_ds.index
        self.set_deeplake_dataset_variables()

    def set_deeplake_dataset_variables(self):
        self.path = self.deeplake_ds.path
        if self.path.startswith("mem://"):
            raise MemoryDatasetCanNotBePickledError

        keys = [
            "base_storage",
            "_read_only",
            # "index",
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
            # "enabled_tensors",
            "is_iteration",
            "_view_base",
            "link_creds",
            "_locked_out",
        ]

        for k in keys:
            setattr(self, k, getattr(self.deeplake_ds, k))

    def merge(
        self,
        target_id: str,
        conflict_resolution: Optional[str] = None,
        delete_removed_tensors: bool = False,
        force: bool = False,
    ):
        raise InvalidOperationError("merge method cannot be called on a Dataset view.")

    def checkout(self, address: str, create: bool = False):
        raise InvalidOperationError("commit method cannot be called on a Dataset view.")

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
        collate_fn: Optional[Callable] = default_collate,
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
            shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch
                (default: ``False``).
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: ``False``)
            retrun_index (bool) Showing wheter Loader needs to return the sample index during iteration.Defaults to True.
            num_workers (int, optional): how many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main process.
                (default: ``0``)
            num_threads (int, optional) number of threads that nedes to be spinned up during data loading. Defaults to None.
                if it is none then we are detecting the hardware concurency count to set.
                Note: We don't set any threading flags (eg. OMP_NUM_THREADS, MKL_NUM_THREADS, etc)
                to get more performant Loader consider of not setting those flags which can affect on 3rd party libraries worflow performance
            transform (Callable, optional) Callable object which is needed to be applyed on each sample on batch. Defaults to None.
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            distributed (nool) flag that is showing wheter Loader needes to work in DDP or not. Defaults to ``False``
            tensors (List[str], optinal) List of tensors thet are participating to in Loadeing process.
                Defaults to ``None`` which means that Loader will fetch samples for all of the tensors
            raw_tensors (List[str], optional) List of the tensors that needs to return raw data instead of decompression.
                Defaults to ``None`` if raw_tensors is None then all the tensors will send decompression data
                E.g raw_tensors['images'] then only the images tensor data will be sent as a row array
            compressed_tensors (List[str], optional) Subset of raw tensors, these will be decompressed by python workers.
            prefetch_factor (int) Number of samples loaded in advance by workers. Defaults to 10
            upcast (bool) floag that is showing wheter we need to upcast object if dtype is not supported this is needed only for
                pytoarch as it is not support all the dtypes. Defaults to True.
            buffer_size (int): The size of the buffer used to shuffle the data in MBs. Defaults to 2048 MB. Increasing the buffer_size will increase the extent of shuffling.
            persistent_workers (bool): If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed once. Defaults to ``False``.

        Raises:
            ImportError: of indra is not installed
        """
        if not _INDRA_INSTALLED:
            raise ImportError(
                "INDRA is not installed. Please install it with `pip install indra`."
            )
        indra_ds = self.indra_ds
        if self.index:
            idx = self.index.values[0].value
            indra_ds = self.indra_ds[idx]

        dataloader = Loader(
            indra_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            return_index=return_index,
            transform_fn=transform,
            num_workers=num_workers,
            num_threads=num_threads,
            collate_fn=collate_fn,
            distributed=distributed,
            tensors=tensors,
            raw_tensors=raw_tensors,
            compressed_tensors=compressed_tensors,
            prefetch_factor=prefetch_factor,
            upcast=upcast,
            primary_tensor=primary_tensor,
            buffer_size=buffer_size,
            persistent_workers=persistent_workers,
        )
        return dataloader

    def __getitem__(
        self,
        item: Union[
            str, int, slice, List[int], Tuple[Union[int, slice, Tuple[int]]], Index
        ],
        is_iteration: bool = False,
    ):
        is_iteration = is_iteration or self.is_iteration
        if isinstance(item, str):
            fullpath = posixpath.join(self.group_index, item)
            enabled_tensors = self.enabled_tensors
            if enabled_tensors is None or fullpath in enabled_tensors:
                tensor = self._get_tensor_from_root(fullpath)
                if tensor is not None:
                    index = self.index
                    if index.is_trivial() and is_iteration == tensor.is_iteration:
                        return tensor
                    return tensor.__getitem__(index, is_iteration=is_iteration)
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
                    if is_iteration and deeplake.constants.SHOW_ITERATION_WARNING:
                        warnings.warn(
                            "Indexing by integer in a for loop, like `for i in range(len(ds)): ... ds[i]` can be quite slow. Use `for i, sample in enumerate(ds)` instead."
                        )
                ret = DeepLakeQueryDataset(
                    deeplake_ds=self.deeplake_ds,
                    indra_ds=self.indra_ds,
                    index=self.index[item],
                )
        else:
            raise InvalidKeyTypeError(item)

        if hasattr(self, "_view_entry"):
            ret._view_entry = self._view_entry
        return ret

    def __len__(self):
        indra_dataset = self.indra_ds
        if self.index:
            idx = self.index.values[0].value
            indra_dataset = self.indra_ds[idx]
        return len(indra_dataset)


class NonlinearQueryView(ViewEntry):
    def __init__(
        self, info: Dict, dataset, source_dataset=None, external: bool = False
    ):
        self.info = info
        self._ds = dataset
        self._src_ds = source_dataset if external else dataset
        self._external = external

    def load(self, verbose=True):
        """Loads the view and returns the :class:`~deeplake.core.dataset.Dataset`.

        Args:
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.

        Returns:
            Dataset: Loaded dataset view.
        """
        ds = self._ds._sub_ds(
            ".queries/" + (self.info.get("path") or self.info["id"]),
            lock=False,
            verbose=False,
        )
        sub_ds_path = ds.path
        if self.virtual:
            ds = ds._get_view(inherit_creds=not self._external)
        ds._view_entry = self
        if verbose:
            log_visualizer_link(sub_ds_path, source_ds_url=self.info["source-dataset"])

        query_str = self.info.get("query")
        indra_ds = ds.query(query_str)

        view = DeepLakeQueryDataset(deeplake_ds=ds, indra_ds=indra_ds)
        return view
