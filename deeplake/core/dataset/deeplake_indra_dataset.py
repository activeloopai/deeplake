import posixpath
from typing import Any, Dict, Optional, Union
from deeplake.client.utils import get_user_name
from deeplake.constants import HUB_CLOUD_DEV_USERNAME
from deeplake.core.dataset import Dataset
from deeplake.client.client import DeepLakeBackendClient
from deeplake.util.bugout_reporter import deeplake_reporter
from deeplake.util.exceptions import (
    InvalidSourcePathError,
    RenameError,
    ReadOnlyModeError,
)
from deeplake.util.link import save_link_creds
from deeplake.util.path import is_hub_cloud_path
from deeplake.util.tag import process_hub_path
from deeplake.util.logging import log_visualizer_link
from deeplake.util.storage import storage_provider_from_hub_path
from warnings import warn
import time
import deeplake
from deeplake.core import tensor
from typing import List, Tuple, Union
from deeplake.core.index import Index
from deeplake.core.index.index import IndexEntry
import numpy as np
from deeplake.core.index import replace_ellipsis_with_slices


class DeepLakeIndraDataset(Dataset):
    def __init__(self, deeplake_ds, indra_ds):
        self.deeplake_ds = deeplake_ds
        self.indra_ds = indra_ds
        self.set_deeplake_dataset_variables()

    def set_deeplake_dataset_variables(self):
        self.path = self.deeplake_ds.path
        if self.path.startswith("mem://"):
            raise MemoryDatasetCanNotBePickledError

        keys = [
            "_read_only",
            "index",
            "group_index",
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
            "enabled_tensors",
            "is_iteration",
            "_view_base",
        ]

        for k in keys:
            setattr(self, k, getattr(self.deeplake_ds, k))
        setattr(self, "link_creds", self.deeplake_ds.link_creds)

    def merge(
        self,
        target_id: str,
        conflict_resolution: Optional[str] = None,
        delete_removed_tensors: bool = False,
        force: bool = False,
    ):
        raise NotImplementedError("This method is not implemented")

    def checkout(self, address: str, create: bool = False):
        raise NotImplementedError("This method is not implemented")

    def _get_tensor_from_root(self, fullpath):
        tensors = self.indra_ds.tensors
        for tensor in tensors:
            if tensor.name == fullpath:
                deeplake_tensor = self.deeplake_ds.__getattr__(fullpath)
                indra_tensor = tensor
                return DeeplakeIndraTensor(deeplake_tensor, indra_tensor)


class DeeplakeIndraTensor(tensor.Tensor):
    def __init__(
        self,
        deeplake_tensor,
        indra_tensors,
        index=None,
        is_iteration: bool = False,
        key: str = None,
    ):
        self.deeplake_tensor = deeplake_tensor
        self.indra_tensors = indra_tensors
        self.is_iteration = is_iteration
        self.set_deeplake_tensor_variables()

        if index:
            self.index = index

        if key:
            self.key = key

    def set_deeplake_tensor_variables(self):
        attrs = [
            "key",
            "dataset",
            "storage",
            "index",
            "version_state",
            "link_creds",
            "_skip_next_setitem",
            "_indexing_history",
        ]

        for k in attrs:
            if hasattr(self.deeplake_tensor, "key"):
                setattr(self, k, getattr(self.deeplake_tensor, k))

        commit_id = self.deeplake_tensor.version_state["commit_id"]

        # if not self.is_iteration and not tensor_exists(
        #     self.key, self.storage, commit_id
        # ):
        #     raise TensorDoesNotExistError(self.key)

        # meta_key = get_tensor_meta_key(self.key, commit_id)
        # meta = self.storage.get_deeplake_object(meta_key, TensorMeta)
        # if not self.pad_tensor and not self.is_iteration:
        #     self.index.validate(self.num_samples)

    def __getitem__(
        self,
        item,
        is_iteration: bool = False,
    ):
        if not isinstance(item, (int, slice, list, tuple, type(Ellipsis), Index)):
            raise InvalidKeyTypeError(item)
        if isinstance(item, tuple) or item is Ellipsis:
            item = replace_ellipsis_with_slices(item, self.ndim)

        key = None
        if hasattr(self, "key"):
            key = self.key

        return DeeplakeIndraTensor(
            self.dataset,
            self.indra_tensors,
            index=self.index[item],
            is_iteration=is_iteration,
            key=key,
        )

    def numpy(self, aslist=True) -> Union[np.ndarray, List[np.ndarray]]:
        idx_len = len(self.index.values)
        idx = self.index.values[0].value

        if isinstance(idx, slice):
            return self.indra_tensors[idx]

        if isinstance(idx, int):
            if idx_len > 1:
                return self.indra_tensors[self.index.values]
            return self.indra_tensors[idx]

        if isinstance(idx, list):
            return self.indra_tensors[idx]

        if isinstance(idx, type(Ellipsis)):
            return self.indra_tensors[:]

    @property
    def num_samples(self):
        return self.indra_tensors.shape[0]

    def can_convert_to_numpy(self):
        if None in self.shape:
            return False
        return True

    @property
    def shape(self):
        idx_len = len(self.index.values)
        idx = self.index.values[0].value

        max_shape = self.indra_tensors.max_shape
        min_shape = self.indra_tensors.min_shape

        first_dim = len(self.indra_tensors)
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or 0
            step = idx.step or 1

            if start < 0:
                start = self.num_samples + start

            if stop < 0:
                stop = self.num_samples + start

            first_dim = (stop - start) // step
        elif isinstance(idx, int):
            first_dim = 1
            if idx_len > 1:
                first_dim = idx_len
        elif isinstance(idx, (list, tuple)):
            first_dim = len(idx)
        elif isinstance(idx, type(Ellipsis)):
            first_dim = len(self.indra_tensors)

        shape = (first_dim,)

        for i, dim_len in enumerate(max_shape):
            if dim_len == min_shape[i]:
                shape += (dim_len,)
            else:
                shape += (None,)
        return shape

    @property
    def ndim(self):
        return len(self.indra_tensors.max_shape)
