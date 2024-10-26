# type: ignore
import os
import uuid
import json
import posixpath
from logging import warning
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Set
from functools import partial

import pathlib
import numpy as np
from time import time, sleep

from jwt import DecodeError
from tqdm import tqdm

import deeplake
from deeplake.client.config import DEEPLAKE_AUTH_TOKEN
from deeplake.core import index_maintenance
from deeplake.core.index.index import IndexEntry
from deeplake.core.link_creds import LinkCreds
from deeplake.core.sample import Sample
from deeplake.core.linked_sample import LinkedSample
from deeplake.util.connect_dataset import connect_dataset_entry
from deeplake.util.downsample import validate_downsampling
from deeplake.util.storage import get_dataset_credentials
from deeplake.util.tag import process_hub_path
from deeplake.util.version_control import (
    save_version_info,
    integrity_check,
    save_commit_info,
    rebuild_version_info,
    _squash_main,
)
from deeplake.util.invalid_view_op import invalid_view_op
from deeplake.util.spinner import spinner
from deeplake.util.iteration_warning import (
    suppress_iteration_warning,
    check_if_iteration,
)
from deeplake.util.tensor_db import parse_runtime_parameters
from deeplake.api.info import load_info
from deeplake.client.log import logger
from deeplake.client.client import DeepLakeBackendClient
from deeplake.constants import (
    FIRST_COMMIT_ID,
    DEFAULT_MEMORY_CACHE_SIZE,
    DEFAULT_LOCAL_CACHE_SIZE,
    MB,
    SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE,
    DEFAULT_READONLY,
    ENV_HUB_DEV_USERNAME,
    QUERY_MESSAGE_MAX_SIZE,
    _INDEX_OPERATION_MAPPING,
)
from deeplake.core.fast_forwarding import ffw_dataset_meta
from deeplake.core.index import Index
from deeplake.core.lock import lock_dataset, unlock_dataset, Lock
from deeplake.core.meta.dataset_meta import DatasetMeta
from deeplake.core.storage import (
    LRUCache,
    S3Provider,
    GCSProvider,
    MemoryProvider,
    LocalProvider,
    AzureProvider,
)
from deeplake.core.tensor import Tensor, create_tensor, delete_tensor
from deeplake.core.version_control.commit_node import CommitNode  # type: ignore
from deeplake.core.version_control.dataset_diff import load_dataset_diff
from deeplake.htype import (
    HTYPE_CONFIGURATIONS,
    UNSPECIFIED,
    verify_htype_key_value,
)
from deeplake.compression import COMPRESSION_ALIASES
from deeplake.integrations import dataset_to_tensorflow
from deeplake.util.bugout_reporter import deeplake_reporter, feature_report_path
from deeplake.util.dataset import try_flushing
from deeplake.util.cache_chain import generate_chain
from deeplake.util.hash import hash_inputs
from deeplake.util.htype import parse_complex_htype
from deeplake.util.link import save_link_creds
from deeplake.util.merge import merge
from deeplake.util.notebook import is_colab
from deeplake.util.path import (
    convert_pathlib_to_string_if_needed,
)
from deeplake.util.scheduling import create_random_split_views
from deeplake.util.logging import log_visualizer_link
from deeplake.util.warnings import always_warn
from deeplake.util.exceptions import (
    CouldNotCreateNewDatasetException,
    InvalidKeyTypeError,
    MemoryDatasetCanNotBePickledError,
    PathNotEmptyException,
    TensorAlreadyExistsError,
    TensorDoesNotExistError,
    TensorGroupDoesNotExistError,
    InvalidTensorNameError,
    InvalidTensorGroupNameError,
    LockedException,
    TensorGroupAlreadyExistsError,
    ReadOnlyModeError,
    RenameError,
    EmptyCommitError,
    DatasetViewSavingError,
    SampleAppendingError,
    DatasetTooLargeToDelete,
    TensorTooLargeToDelete,
    GroupInfoNotSupportedError,
    TokenPermissionError,
    CheckoutError,
    DatasetCorruptError,
    BadRequestException,
    SampleAppendError,
    SampleExtendError,
    DatasetHandlerError,
)
from deeplake.util.keys import (
    dataset_exists,
    get_dataset_info_key,
    get_dataset_meta_key,
    tensor_exists,
    get_queries_key,
    get_queries_lock_key,
    get_sample_id_tensor_key,
    get_sample_info_tensor_key,
    get_sample_shape_tensor_key,
    get_downsampled_tensor_key,
    filter_name,
    get_dataset_linked_creds_key,
    get_tensor_meta_key,
    get_chunk_id_encoder_key,
    get_tensor_tile_encoder_key,
    get_creds_encoder_key,
)

from deeplake.util.path import get_path_from_storage, relpath
from deeplake.util.remove_cache import get_base_storage
from deeplake.util.diff import get_all_changes_string, get_changes_and_messages
from deeplake.util.version_control import (
    auto_checkout,
    checkout,
    delete_branch,
    commit,
    current_commit_has_change,
    load_meta,
    warn_node_checkout,
    load_version_info,
    save_version_info,
    replace_head,
    reset_and_checkout,
)
from deeplake.util.pretty_print import summary_dataset
from deeplake.core.dataset.view_entry import ViewEntry
from deeplake.core.dataset.invalid_view import InvalidView
from deeplake.hooks import dataset_read
from collections import defaultdict
from itertools import chain
import warnings
import jwt

_LOCKABLE_STORAGES = {S3Provider, GCSProvider, AzureProvider, LocalProvider}


def _load_tensor_metas(dataset):
    meta_keys = [
        get_tensor_meta_key(key, dataset.version_state["commit_id"])
        for key in dataset.meta.tensors
    ]
    for _ in dataset.storage.get_items(meta_keys):
        pass
    dataset._tensors()  # access all tensors to set chunk engines


class Dataset:
    def __init__(
        self,
        storage: LRUCache,
        index: Optional[Index] = None,
        group_index: str = "",
        read_only: Optional[bool] = None,
        public: Optional[bool] = False,
        token: Optional[str] = None,
        org_id: Optional[str] = None,
        verbose: bool = True,
        version_state: Optional[Dict[str, Any]] = None,
        path: Optional[Union[str, pathlib.Path]] = None,
        address: Optional[str] = None,
        is_iteration: bool = False,
        link_creds=None,
        pad_tensors: bool = False,
        lock_enabled: bool = True,
        lock_timeout: Optional[int] = 0,
        enabled_tensors: Optional[List[str]] = None,
        view_base: Optional["Dataset"] = None,
        libdeeplake_dataset=None,
        index_params: Optional[Dict[str, Union[int, str]]] = None,
        dataset_creds_key: Optional[str] = None,
        dataset_creds_key_org_id: Optional[str] = None,
        dataset_creds_key_token: Optional[str] = None,
        **kwargs,
    ):
        """Initializes a new or existing dataset.

        Args:
            storage (LRUCache): The storage provider used to access the dataset.
            index (Index, Optional): The Index object restricting the view of this dataset's tensors.
            group_index (str): Name of the group this dataset instance represents.
            read_only (bool, Optional): Opens dataset in read only mode if this is passed as True. Defaults to False.
                Datasets stored on Deep Lake cloud that your account does not have write access to will automatically open in read mode.
            public (bool, Optional): Applied only if storage is Deep Lake cloud storage and a new Dataset is being created. Defines if the dataset will have public access.
            token (str, Optional): Activeloop token, used for fetching credentials for Deep Lake datasets. This is Optional, tokens are normally autogenerated.
            org_id (str, Optional): Organization id to be used for enabling high-performance features. Only applicable for local datasets.
            verbose (bool): If ``True``, logs will be printed. Defaults to True.
            version_state (Dict[str, Any], Optional): The version state of the dataset, includes commit_id, commit_node, branch, branch_commit_map and commit_node_map.
            path (str, pathlib.Path): The path to the dataset.
            address (Optional[str]): The version address of the dataset.
            is_iteration (bool): If this Dataset is being used as an iterator.
            link_creds (LinkCreds, Optional): The LinkCreds object used to access tensors that have external data linked to them.
            pad_tensors (bool): If ``True``, shorter tensors will be padded to the length of the longest tensor.
            **kwargs: Passing subclass variables through without errors.
            lock_enabled (bool): Whether the dataset should be locked for writing. Only applicable for S3, Deep Lake storage and GCS datasets. No effect if read_only=True.
            lock_timeout (int): How many seconds to wait for a lock before throwing an exception. If None, wait indefinitely
            enabled_tensors (List[str], Optional): List of tensors that are enabled in this view. By default all tensors are enabled.
            view_base (Optional["Dataset"]): Base dataset of this view.
            libdeeplake_dataset : The libdeeplake dataset object corresponding to this dataset.
            index_params: (Dict[str, Union[int, str]] Optional) VDB index parameter. Defaults to ``None.``
            dataset_creds_key: (str, Optional) The key to use for fetching dataset credentials. Defaults to ``None``.
            dataset_creds_key_org_id: (str, Optional) If dataset_creds_key is set, the org_id the key lives in
            dataset_creds_key_token: (str, Optional) If dataset_creds_key is set, the token used to access the credentials

        Raises:
            ValueError: If an existing local path is given, it must be a directory.
            ImproperDatasetInitialization: Exactly one argument out of 'path' and 'storage' needs to be specified.
                This is raised if none of them are specified or more than one are specifed.
            InvalidHubPathException: If a Deep Lake cloud path (path starting with `hub://`) is specified and it isn't of the form `hub://username/datasetname`.
            AuthorizationException: If a Deep Lake cloud path (path starting with `hub://`) is specified and the user doesn't have access to the dataset.
            PathNotEmptyException: If the path to the dataset doesn't contain a Deep Lake dataset and is also not empty.
            LockedException: If read_only is False but the dataset is locked for writing by another machine.
            ReadOnlyModeError: If read_only is False but write access is not available.
        """
        d: Dict[str, Any] = {}
        d["_client"] = d["ds_name"] = None
        # uniquely identifies dataset
        d["path"] = convert_pathlib_to_string_if_needed(path) or get_path_from_storage(
            storage
        )
        d["storage"] = storage
        d["_read_only_error"] = read_only is False
        d["base_storage"] = get_base_storage(storage)
        d["_read_only"] = d["base_storage"].read_only
        d["_locked_out"] = False  # User requested write access but was denied
        d["is_iteration"] = is_iteration
        d["is_first_load"] = version_state is None
        d["_is_filtered_view"] = False
        d["index"] = index or Index()
        d["group_index"] = group_index
        d["_token"] = token
        d["org_id"] = org_id
        d["public"] = public
        d["verbose"] = verbose
        d["version_state"] = version_state or {}
        d["link_creds"] = link_creds
        d["enabled_tensors"] = enabled_tensors
        d["libdeeplake_dataset"] = libdeeplake_dataset
        d["index_params"] = index_params
        d["_info"] = None
        d["_ds_diff"] = None
        d["_view_id"] = str(uuid.uuid4())
        d["_view_base"] = view_base
        d["_view_use_parent_commit"] = False
        d["_update_hooks"] = {}
        d["_commit_hooks"] = {}
        d["_checkout_hooks"] = {}
        d["_parent_dataset"] = None
        d["_pad_tensors"] = pad_tensors
        d["_locking_enabled"] = lock_enabled
        d["_lock_timeout"] = lock_timeout
        d["_temp_tensors"] = []
        d["_vc_info_updated"] = True
        d["_query_string"] = None
        d["dataset_creds_key"] = dataset_creds_key
        d["dataset_creds_key_org_id"] = dataset_creds_key_org_id
        d["dataset_creds_key_token"] = dataset_creds_key_token
        dct = self.__dict__
        dct.update(d)

        try:
            self._set_derived_attributes(address=address)
        except LockedException:
            raise LockedException(
                "This dataset cannot be open for writing as it is locked by another machine. Try loading the dataset with `read_only=True`."
            )
        except ReadOnlyModeError as e:
            raise ReadOnlyModeError(
                "This dataset cannot be open for writing as you don't have permissions. Try loading the dataset with `read_only=True."
            )
        dct["enabled_tensors"] = (
            set(self._resolve_tensor_list(enabled_tensors, root=True))
            if enabled_tensors
            else None
        )
        self._first_load_init()
        self._initial_autoflush: List[bool] = (
            []
        )  # This is a stack to support nested with contexts

        self._indexing_history: List[int] = []

        if not self.read_only:
            for temp_tensor in self._temp_tensors:
                self._delete_tensor(temp_tensor, large_ok=True)
            self._temp_tensors = []

        if self.is_first_load:
            _load_tensor_metas(self)

    def _lock_lost_handler(self):
        """This is called when lock is acquired but lost later on due to slow update."""
        self.read_only = True
        if self.verbose:
            always_warn(
                "Unable to update dataset lock as another machine has locked it for writing or deleted / overwriten it. Switching to read only mode."
            )
        self._locked_out = True

    def __enter__(self):
        self._initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        autoflush = self._initial_autoflush.pop()
        if not self._read_only and autoflush:
            if self._vc_info_updated:
                self._flush_vc_info()
            spinner(self.storage.flush)()
        self.storage.autoflush = autoflush

    def maybe_flush(self):
        if not self._read_only:
            if self.storage.autoflush:
                if self._vc_info_updated:
                    self._flush_vc_info()
                self.storage.flush()

    @property
    def username(self) -> str:
        if not self.token:
            return "public"

        try:
            return jwt.decode(self.token, options={"verify_signature": False})["id"]
        except:
            return "public"

    @property
    def num_samples(self) -> int:
        """Returns the length of the smallest tensor.
        Ignores any applied indexing and returns the total length.
        """
        return min(
            map(
                len,
                filter(
                    lambda t: t.key not in self.meta.hidden_tensors,
                    self.version_state["full_tensors"].values(),
                ),
            ),
            default=0,
        )

    @property
    def meta(self) -> DatasetMeta:
        """Returns the metadata of the dataset."""
        return self.version_state["meta"]

    @property
    def client(self):
        """Returns the client of the dataset."""
        return self._client

    def __len__(self, warn: bool = True):
        """Returns the length (number of rows) of the shortest tensor in the dataset."""
        tensor_lengths = [len(tensor) for tensor in self.tensors.values()]
        pad_tensors = self._pad_tensors
        if (
            warn
            and not pad_tensors
            and min(tensor_lengths, default=0) != max(tensor_lengths, default=0)
        ):
            warning(
                "The length of tensors in the dataset is different. The len(ds) returns the length of the "
                "smallest tensor in the dataset. If you want the length of the longest tensor in the dataset use "
                "ds.max_len."
            )
        length_fn = max if pad_tensors else min
        return length_fn(tensor_lengths, default=0)

    @property
    def max_len(self):
        """Returns the length (number of rows) of the longest tensor in the dataset."""
        return (
            max([len(tensor) for tensor in self.tensors.values()])
            if self.tensors
            else 0
        )

    @property
    def min_len(self):
        """Returns the length (number of rows) of the shortest tensor in the dataset."""
        return (
            min([len(tensor) for tensor in self.tensors.values()])
            if self.tensors
            else 0
        )

    def __getstate__(self) -> Dict[str, Any]:
        """Returns a dict that can be pickled and used to restore this dataset.

        Note:
            Pickling a dataset does not copy the dataset, it only saves attributes that can be used to restore the dataset.
            If you pickle a local dataset and try to access it on a machine that does not have the data present, the dataset will not work.
        """
        if self.path.startswith("mem://"):
            raise MemoryDatasetCanNotBePickledError
        keys = [
            "path",
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
            "_lock_timeout",
            "enabled_tensors",
            "is_iteration",
            "index_params",
        ]
        state = {k: getattr(self, k) for k in keys}
        state["link_creds"] = self.link_creds
        return state

    def __setstate__(self, state: Dict[str, Any]):
        """Restores dataset from a pickled state.

        Args:
            state (dict): The pickled state used to restore the dataset.
        """
        state["is_first_load"] = True
        state["_info"] = None
        state["_read_only_error"] = False
        state["_initial_autoflush"] = []
        state["_ds_diff"] = None
        state["_view_base"] = None
        state["_update_hooks"] = {}
        state["_commit_hooks"] = {}
        state["_checkout_hooks"] = {}
        state["_client"] = state["org_id"] = state["ds_name"] = None
        state["_temp_tensors"] = []
        state["libdeeplake_dataset"] = None
        state["_vc_info_updated"] = False
        state["_locked_out"] = False
        self.__dict__.update(state)
        self.__dict__["base_storage"] = get_base_storage(self.storage)
        # clear cache while restoring
        self.storage.clear_cache_without_flush()
        self._set_derived_attributes(verbose=False)
        self._indexing_history = []

    def _reload_version_state(self):
        version_state = self.version_state
        # share version state if at HEAD
        if (
            not self._view_use_parent_commit
            and self._view_base
            and version_state["commit_node"].is_head_node
        ):
            uid = self._view_id

            def commit_hook():
                del self._view_base._commit_hooks[uid]
                del self._view_base._checkout_hooks[uid]
                del self._view_base._update_hooks[uid]
                self._view_use_parent_commit = True
                self._reload_version_state()

            def checkout_hook():
                del self._view_base._commit_hooks[uid]
                del self._view_base._checkout_hooks[uid]
                del self._view_base._update_hooks[uid]
                self.__class__ = InvalidView
                self.__init__(reason="checkout")

            def update_hook():
                del self._view_base._commit_hooks[uid]
                del self._view_base._checkout_hooks[uid]
                del self._view_base._update_hooks[uid]
                self.__class__ = InvalidView
                self.__init__(reason="update")

            if not self.is_iteration:
                self._view_base._commit_hooks[uid] = commit_hook
                self._view_base._checkout_hooks[uid] = checkout_hook
                self._view_base._update_hooks[uid] = update_hook
            return version_state
        vs_copy = {}
        vs_copy["branch"] = version_state["branch"]
        # share branch_commit_map and commit_node_map
        vs_copy["branch_commit_map"] = version_state["branch_commit_map"]
        vs_copy["commit_node_map"] = version_state["commit_node_map"]
        commit_node = version_state["commit_node"]
        if self._view_use_parent_commit:
            vs_copy["commit_node"] = commit_node.parent
        else:
            vs_copy["commit_node"] = commit_node
        vs_copy["commit_id"] = vs_copy["commit_node"].commit_id
        vs_copy["tensor_names"] = version_state["tensor_names"].copy()
        vs_copy["meta"] = DatasetMeta()
        vs_copy["meta"].__setstate__(version_state["meta"].__getstate__())
        self.version_state = vs_copy
        vs_copy["full_tensors"] = {
            key: Tensor(key, self) for key in version_state["full_tensors"]
        }
        self._view_base = None

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
                ret = self.__class__(
                    storage=self.storage,
                    index=self.index,
                    group_index=posixpath.join(self.group_index, item),
                    read_only=self.read_only,
                    token=self._token,
                    verbose=False,
                    version_state=self.version_state,
                    path=self.path,
                    is_iteration=is_iteration,
                    link_creds=self.link_creds,
                    pad_tensors=self._pad_tensors,
                    enabled_tensors=self.enabled_tensors,
                    view_base=self._view_base or self,
                    libdeeplake_dataset=self.libdeeplake_dataset,
                    index_params=self.index_params,
                )
            elif "/" in item:
                splt = posixpath.split(item)
                ret = (
                    self[splt[1]]
                    if splt[0] == self.group_index
                    else self[splt[0]][splt[1]]
                )
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
                for x in enabled_tensors:
                    enabled_tensors.extend(
                        self[relpath(x, self.group_index)].meta.links.keys()
                    )

                ret = self.__class__(
                    storage=self.storage,
                    index=self.index,
                    group_index=self.group_index,
                    read_only=self._read_only,
                    token=self._token,
                    verbose=False,
                    version_state=self.version_state,
                    path=self.path,
                    is_iteration=is_iteration,
                    link_creds=self.link_creds,
                    pad_tensors=self._pad_tensors,
                    enabled_tensors=enabled_tensors,
                    view_base=self._view_base or self,
                    libdeeplake_dataset=self.libdeeplake_dataset,
                    index_params=self.index_params,
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

                ret = self.__class__(
                    storage=self.storage,
                    index=self.index[item],
                    group_index=self.group_index,
                    read_only=self._read_only,
                    token=self._token,
                    verbose=False,
                    version_state=self.version_state,
                    path=self.path,
                    is_iteration=is_iteration,
                    link_creds=self.link_creds,
                    pad_tensors=self._pad_tensors,
                    enabled_tensors=self.enabled_tensors,
                    view_base=self._view_base or self,
                    libdeeplake_dataset=self.libdeeplake_dataset,
                    index_params=self.index_params,
                )
        else:
            raise InvalidKeyTypeError(item)
        if hasattr(self, "_view_entry"):
            ret._view_entry = self._view_entry
        return ret

    def __setitem__(self, item: str, value: Any):
        if not isinstance(item, str):
            raise TypeError("Datasets do not support item assignment")
        tensor = self[item]
        if isinstance(tensor, Dataset):
            raise TypeError("Tensor groups do not support item assignment")
        tensor.index = Index()
        tensor._update(self.index, value)

    @invalid_view_op
    def create_tensor(
        self,
        name: str,
        htype: str = UNSPECIFIED,
        dtype: Union[str, np.dtype] = UNSPECIFIED,
        sample_compression: str = UNSPECIFIED,
        chunk_compression: str = UNSPECIFIED,
        hidden: bool = False,
        create_sample_info_tensor: bool = True,
        create_shape_tensor: bool = True,
        create_id_tensor: bool = True,
        verify: bool = True,
        exist_ok: bool = False,
        verbose: bool = True,
        downsampling: Optional[Tuple[int, int]] = None,
        tiling_threshold: Optional[int] = None,
        **kwargs,
    ):
        """Creates a new tensor in the Deep Lake dataset. Specifying the tensor's htype is highly recommended for complex data such as images, video, dicom, text, json, etc.
        Specifying htype is not necessary for simple numeric date such as arrays and scalars.

        Examples:
            >>> # Create dataset
            >>> ds = deeplake.empty("path/to/dataset")

            >>> # Create tensors
            >>> with ds:
            >>>     ds.create_tensor("images", htype="image", sample_compression="jpg")
            >>>     ds.create_tensor("videos", htype="video", sample_compression="mp4")
            >>>     ds.create_tensor("data")
            >>>     ds.create_tensor("point_clouds", htype="point_cloud")
            >>>
            >>>     # Append data
            >>>     ds.images.append(np.ones((400, 400, 3), dtype='uint8'))
            >>>     ds.videos.append(deeplake.read("videos/sample_video.mp4"))
            >>>     ds.data.append(np.zeros((100, 100, 2)))

        Args:
            name (str): The name of the tensor to be created.
            htype (str):
                - The class of data for the tensor. Specifying ``htype`` is highly recommended for complex data such as images, video, dicom, text, json, etc. Specifying htype is not necessary for simple numeric date such as arrays and scalars.
                - The defaults for other parameters are determined in terms of this value, thus improving error checking and visualization.
                - For example, ``htype="image"`` would have ``dtype`` default to ``uint8``.
                - These defaults can be overridden by explicitly passing any of the other parameters to this function.
                - May also modify the defaults for other parameters.
            dtype (str): Optionally override this tensor's ``dtype``. All subsequent samples are required to have this ``dtype``. Specifying ``dtype`` is typically not necessary unless you are dealing with mixed numerical data in a single tensor (column).
            sample_compression (str): All samples will be compressed in the provided format. If ``None``, samples are uncompressed. For ``link[]`` tensors, ``sample_compression`` is used only for optimizing dataset views.
            chunk_compression (str): All chunks will be compressed in the provided format. If ``None``, chunks are uncompressed. For ``link[]`` tensors, ``chunk_compression`` is used only for optimizing dataset views.
            hidden (bool): If ``True``, the tensor will be hidden from ds.tensors but can still be accessed via ``ds[tensor_name]``.
            create_sample_info_tensor (bool): If ``True``, meta data of individual samples will be saved in a hidden tensor. This data can be accessed via :attr:`tensor[i].sample_info <deeplake.core.tensor.Tensor.sample_info>`.
            create_shape_tensor (bool): If ``True``, an associated tensor containing shapes of each sample will be created.
            create_id_tensor (bool): If ``True``, an associated tensor containing unique ids for each sample will be created. This is useful for merge operations.
            verify (bool): Valid only for link htypes. If ``True``, all links will be verified before they are added to the tensor.
                If ``False``, links will be added without verification but note that ``create_shape_tensor`` and ``create_sample_info_tensor`` will be set to ``False``.
            exist_ok (bool): If ``True``, the group is created if it does not exist. if ``False``, an error is raised if the group already exists.
            verbose (bool): Shows warnings if ``True``.
            downsampling (tuple[int, int]): If not ``None``, the tensor will be downsampled by the provided factors. For example, ``(2, 5)`` will downsample the tensor by a factor of 2 in both dimensions and create 5 layers of downsampled tensors.
                Only support for image and mask htypes.
            tiling_threshold (Optional, int): In bytes. Tiles large images if their size exceeds this threshold. Set to -1 to disable tiling.
            **kwargs:
                - ``htype`` defaults can be overridden by passing any of the compatible parameters.
                - To see all htypes and their correspondent arguments, check out :ref:`Htypes`.

        Returns:
            Tensor: The new tensor, which can be accessed by ``dataset[name]`` or ``dataset.name``.

        Raises:
            TensorAlreadyExistsError: If the tensor already exists and ``exist_ok`` is ``False``.
            TensorGroupAlreadyExistsError: Duplicate tensor groups are not allowed.
            InvalidTensorNameError: If ``name`` is in dataset attributes.
            NotImplementedError: If trying to override ``chunk_compression``.
            TensorMetaInvalidHtype: If invalid htype is specified.
            ValueError: If an illegal argument is specified.
        """

        deeplake_reporter.feature_report(
            feature_name="create_tensor",
            parameters={
                "name": name,
                "htype": htype,
                "dtype": dtype,
                "sample_compression": sample_compression,
                "chunk_compression": chunk_compression,
            },
        )

        return self._create_tensor(
            name,
            htype,
            dtype,
            sample_compression,
            chunk_compression,
            hidden,
            create_sample_info_tensor,
            create_shape_tensor,
            create_id_tensor,
            verify,
            exist_ok,
            verbose,
            downsampling,
            tiling_threshold=tiling_threshold,
            **kwargs,
        )

    @invalid_view_op
    def _create_tensor(
        self,
        name: str,
        htype: str = UNSPECIFIED,
        dtype: Union[str, np.dtype] = UNSPECIFIED,
        sample_compression: str = UNSPECIFIED,
        chunk_compression: str = UNSPECIFIED,
        hidden: bool = False,
        create_sample_info_tensor: bool = True,
        create_shape_tensor: bool = True,
        create_id_tensor: bool = True,
        verify: bool = True,
        exist_ok: bool = False,
        verbose: bool = True,
        downsampling: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        # if not the head node, checkout to an auto branch that is newly created
        auto_checkout(self)

        name = filter_name(name, self.group_index)
        key = self.version_state["tensor_names"].get(name)
        is_sequence, is_link, htype = parse_complex_htype(htype)
        if key:
            if not exist_ok:
                raise TensorAlreadyExistsError(name)
            tensor = self.root[key]
            current_config = tensor._config
            new_config = {
                "htype": htype,
                "dtype": dtype,
                "sample_compression": COMPRESSION_ALIASES.get(
                    sample_compression, sample_compression
                ),
                "chunk_compression": COMPRESSION_ALIASES.get(
                    chunk_compression, chunk_compression
                ),
                "hidden": hidden,
                "is_link": is_link,
                "is_sequence": is_sequence,
            }
            base_config = HTYPE_CONFIGURATIONS.get(htype, {}).copy()
            for key in new_config:
                if new_config[key] == UNSPECIFIED:
                    new_config[key] = base_config.get(key) or UNSPECIFIED
            if current_config != new_config:
                raise ValueError(
                    f"Tensor {name} already exists with different configuration. "
                    f"Current config: {current_config}. "
                    f"New config: {new_config}"
                )
            return tensor
        elif name in self.version_state["full_tensors"]:
            key = f"{name}_{uuid.uuid4().hex[:4]}"
        else:
            key = name

        if name in self._groups:
            raise TensorGroupAlreadyExistsError(name)

        tensor_name = posixpath.split(name)[1]
        if not tensor_name or tensor_name in dir(self):
            raise InvalidTensorNameError(tensor_name)

        downsampling_factor, number_of_layers = validate_downsampling(downsampling)
        kwargs["is_sequence"] = kwargs.get("is_sequence") or is_sequence
        kwargs["is_link"] = kwargs.get("is_link") or is_link
        kwargs["verify"] = verify
        if kwargs["is_link"] and not kwargs["verify"]:
            create_shape_tensor = False
            create_sample_info_tensor = False

        if not self._is_root():
            return self.root._create_tensor(
                name=key,
                htype=htype,
                dtype=dtype,
                sample_compression=sample_compression,
                chunk_compression=chunk_compression,
                hidden=hidden,
                create_sample_info_tensor=create_sample_info_tensor,
                create_shape_tensor=create_shape_tensor,
                create_id_tensor=create_id_tensor,
                exist_ok=exist_ok,
                downsampling=downsampling,
                **kwargs,
            )

        if "/" in name:
            self._create_group(posixpath.split(name)[0])

        # Seperate meta and info

        htype_config = HTYPE_CONFIGURATIONS.get(htype, {}).copy()
        info_keys = htype_config.pop("_info", [])
        info_kwargs = {}
        meta_kwargs = {}
        for k, v in kwargs.items():
            if k in info_keys:
                verify_htype_key_value(htype, k, v)
                info_kwargs[k] = v
            else:
                meta_kwargs[k] = v

        # Set defaults
        for k in info_keys:
            if k not in info_kwargs:
                if k == "class_names":
                    info_kwargs[k] = htype_config[k].copy()
                else:
                    info_kwargs[k] = htype_config[k]

        create_tensor(
            key,
            self.storage,
            htype=htype,
            dtype=dtype,
            sample_compression=sample_compression,
            chunk_compression=chunk_compression,
            version_state=self.version_state,
            hidden=hidden,
            overwrite=True,
            **meta_kwargs,
        )
        meta: DatasetMeta = self.meta
        ffw_dataset_meta(meta)
        meta.add_tensor(name, key, hidden=hidden)
        tensor = Tensor(key, self)  # type: ignore
        tensor.meta.name = name
        self.version_state["full_tensors"][key] = tensor
        self.version_state["tensor_names"][name] = key
        if info_kwargs:
            tensor.info.update(info_kwargs)
        self.storage.maybe_flush()
        if create_sample_info_tensor and htype in (
            "image",
            "audio",
            "video",
            "dicom",
            "point_cloud",
            "mesh",
            "nifti",
            "segment_mask",
        ):
            self._create_sample_info_tensor(name)
        if create_shape_tensor and htype not in ("text", "json", "tag"):
            self._create_sample_shape_tensor(name, htype=htype)
        if create_id_tensor:
            self._create_sample_id_tensor(name)
        if downsampling:
            downsampling_htypes = {
                "image",
                "image.rgb",
                "image.gray",
                "binary_mask",
                "segment_mask",
            }
            if htype not in downsampling_htypes:
                warnings.warn(
                    f"Downsampling is only supported for tensor with htypes {downsampling_htypes}, got {htype}. Skipping downsampling."
                )
            else:
                self._create_downsampled_tensor(
                    name,
                    htype,
                    dtype,
                    sample_compression,
                    chunk_compression,
                    meta_kwargs,
                    downsampling_factor,
                    number_of_layers,
                )
        return tensor

    def _create_sample_shape_tensor(self, tensor: str, htype: str):
        shape_tensor = get_sample_shape_tensor_key(tensor)
        self._create_tensor(
            shape_tensor,
            dtype="int64",
            hidden=True,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            max_chunk_size=SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE,
        )
        if htype == "list":
            extend_f = "extend_len"
            update_f = "update_len"
        else:
            extend_f = "extend_shape"
            update_f = "update_shape"
        self._link_tensors(
            tensor,
            shape_tensor,
            extend_f=extend_f,
            update_f=update_f,
            flatten_sequence=True,
        )

    def _create_sample_id_tensor(self, tensor: str):
        id_tensor = get_sample_id_tensor_key(tensor)
        self._create_tensor(
            id_tensor,
            hidden=True,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
        )
        self._link_tensors(
            tensor,
            id_tensor,
            extend_f="extend_id",
            flatten_sequence=False,
        )

    def _create_sample_info_tensor(self, tensor: str):
        sample_info_tensor = get_sample_info_tensor_key(tensor)
        self._create_tensor(
            sample_info_tensor,
            htype="json",
            max_chunk_size=SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE,
            hidden=True,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
        )
        self._link_tensors(
            tensor,
            sample_info_tensor,
            "extend_info",
            "update_info",
            flatten_sequence=True,
        )

    def _create_downsampled_tensor(
        self,
        tensor: str,
        htype: str,
        dtype: Union[str, np.dtype],
        sample_compression: str,
        chunk_compression: str,
        meta_kwargs: Dict[str, Any],
        downsampling_factor: int,
        number_of_layers: int,
    ):
        downsampled_tensor = get_downsampled_tensor_key(tensor, downsampling_factor)
        if number_of_layers == 1:
            downsampling = None
        else:
            downsampling = (downsampling_factor, number_of_layers - 1)
        meta_kwargs = meta_kwargs.copy()
        meta_kwargs.pop("is_link", None)
        new_tensor = self._create_tensor(
            downsampled_tensor,
            htype=htype,
            dtype=dtype,
            sample_compression=sample_compression,
            chunk_compression=chunk_compression,
            hidden=True,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            downsampling=downsampling,
            **meta_kwargs,
        )
        new_tensor.info.downsampling_factor = downsampling_factor
        self._link_tensors(
            tensor,
            downsampled_tensor,
            extend_f=f"extend_downsample",
            update_f=f"update_downsample",
            flatten_sequence=True,
        )

    def _hide_tensor(self, tensor: str):
        self._tensors()[tensor].meta.set_hidden(True)
        self.meta._hide_tensor(tensor)
        self.storage.maybe_flush()

    @invalid_view_op
    def delete_tensor(self, name: str, large_ok: bool = False):
        """Delete a tensor from the dataset.

        Examples:

            >>> ds.delete_tensor("images/cats")

        Args:
            name (str): The name of tensor to be deleted.
            large_ok (bool): Delete tensors larger than 1 GB. Disabled by default.

        Returns:
            None

        Raises:
            TensorDoesNotExistError: If tensor of name ``name`` does not exist in the dataset.
            TensorTooLargeToDelete: If the tensor is larger than 1 GB and ``large_ok`` is ``False``.
        """

        deeplake_reporter.feature_report(
            feature_name="delete_tensor",
            parameters={"name": name, "large_ok": large_ok},
        )

        return self._delete_tensor(name, large_ok)

    @invalid_view_op
    def _delete_tensor(self, name: str, large_ok: bool = False):
        auto_checkout(self)

        name = filter_name(name, self.group_index)
        key = self.version_state["tensor_names"].get(name)

        if not key:
            raise TensorDoesNotExistError(name)

        if not tensor_exists(key, self.storage, self.version_state["commit_id"]):
            raise TensorDoesNotExistError(name)

        if not self._is_root():
            return self.root._delete_tensor(name, large_ok)

        if not large_ok:
            chunk_engine = self.version_state["full_tensors"][key].chunk_engine
            size_approx = chunk_engine.num_samples * chunk_engine.min_chunk_size
            if size_approx > deeplake.constants.DELETE_SAFETY_SIZE:
                raise TensorTooLargeToDelete(name)

        with self:
            meta = self.meta
            key = self.version_state["tensor_names"].pop(name)
            if key not in meta.hidden_tensors:
                tensor_diff = Tensor(key, self).chunk_engine.commit_diff
                # if tensor was created in this commit, there's no diff for deleting it.
                if not tensor_diff.created:
                    self._dataset_diff.tensor_deleted(name)
            delete_tensor(key, self)
            self.version_state["full_tensors"].pop(key)
            ffw_dataset_meta(meta)
            meta.delete_tensor(name)

        for t_name in [
            func(name)
            for func in (
                get_sample_id_tensor_key,
                get_sample_info_tensor_key,
                get_sample_shape_tensor_key,
            )
        ]:
            t_key = self.meta.tensor_names.get(t_name)
            if t_key and tensor_exists(
                t_key, self.storage, self.version_state["commit_id"]
            ):
                self._delete_tensor(t_name, large_ok=True)

        self.storage.flush()

    @invalid_view_op
    def delete_group(self, name: str, large_ok: bool = False):
        """Delete a tensor group from the dataset.

        Examples:
            >>> ds.delete_group("images/dogs")

        Args:
            name (str): The name of tensor group to be deleted.
            large_ok (bool): Delete tensor groups larger than 1 GB. Disabled by default.

        Returns:
            None

        Raises:
            TensorGroupDoesNotExistError: If tensor group of name ``name`` does not exist in the dataset.
        """

        deeplake_reporter.feature_report(
            feature_name="delete_group",
            parameters={"name": name, "large_ok": large_ok},
        )

        return self._delete_group(name, large_ok)

    @invalid_view_op
    def _delete_group(self, name: str, large_ok: bool = False):
        auto_checkout(self)

        full_path = filter_name(name, self.group_index)

        if full_path not in self._groups:
            raise TensorGroupDoesNotExistError(name)

        if not self._is_root():
            return self.root._delete_group(full_path, large_ok)

        if not large_ok:
            size_approx = self[name].size_approx()
            if size_approx > deeplake.constants.DELETE_SAFETY_SIZE:
                logger.info(
                    f"Group {name} was too large to delete. Try again with large_ok=True."
                )
                return

        with self:
            meta = self.meta
            ffw_dataset_meta(meta)
            tensors = [
                posixpath.join(name, tensor)
                for tensor in self[name]._all_tensors_filtered(include_hidden=True)
            ]
            meta.delete_group(name)
            for tensor in tensors:
                key = self.version_state["tensor_names"].pop(tensor)
                if key not in meta.hidden_tensors:
                    tensor_diff = Tensor(key, self).chunk_engine.commit_diff
                    # if tensor was created in this commit, there's no diff for deleting it.
                    if not tensor_diff.created:
                        self._dataset_diff.tensor_deleted(name)
                delete_tensor(key, self)
                self.version_state["full_tensors"].pop(key)

        self.storage.maybe_flush()

    @invalid_view_op
    def create_tensor_like(
        self, name: str, source: "Tensor", unlink: bool = False
    ) -> "Tensor":
        """Creates a tensor with the same properties as ``source``. No samples or data copied, other than the tensor meta/info.

        Examples:
            >>> ds.create_tensor_like("cats", ds["images"])

        Args:
            name (str): Name for the new tensor.
            source (Tensor): Tensor who's meta/info will be copied. May or may not be contained in the same dataset.
            unlink (bool): Whether to unlink linked tensors.

        Returns:
            Tensor: New Tensor object.
        """

        deeplake_reporter.feature_report(
            feature_name="create_tensor_like",
            parameters={"name": name, "unlink": unlink},
        )

        info = source.info.__getstate__().copy()
        meta = source.meta.__getstate__().copy()
        if unlink:
            meta["is_link"] = False
        del meta["min_shape"]
        del meta["max_shape"]
        del meta["length"]
        del meta["version"]
        del meta["name"]
        del meta["links"]
        if "vdb_indexes" in meta:
            del meta["vdb_indexes"]
        meta["dtype"] = np.dtype(meta["typestr"]) if meta["typestr"] else meta["dtype"]

        destination_tensor = self._create_tensor(
            name,
            verbose=False,
            create_id_tensor=bool(source._sample_id_tensor),
            create_shape_tensor=bool(source._sample_shape_tensor),
            create_sample_info_tensor=bool(source._sample_info_tensor),
            **meta,
        )
        destination_tensor.info.update(info)
        return destination_tensor

    def _rename_tensor(self, name, new_name):
        tensor = self[name]
        tensor.meta.name = new_name
        tensor.meta.is_dirty = True
        key = self.version_state["tensor_names"].pop(name)
        meta = self.meta
        if key not in meta.hidden_tensors:
            tensor_diff = tensor.chunk_engine.commit_diff
            # if tensor was created in this commit, tensor name has to be updated without adding it to diff.
            if not tensor_diff.created:
                self._dataset_diff.tensor_renamed(name, new_name)
        self.version_state["tensor_names"][new_name] = key
        ffw_dataset_meta(meta)
        meta.rename_tensor(name, new_name)

        for func in (
            get_sample_id_tensor_key,
            get_sample_info_tensor_key,
            get_sample_shape_tensor_key,
        ):
            t_old, t_new = map(func, (name, new_name))
            t_key = self.meta.tensor_names.get(t_old)
            if t_key and tensor_exists(
                t_key, self.storage, self.version_state["commit_id"]
            ):
                self._rename_tensor(t_old, t_new)

        return tensor

    def rename_tensor(self, name: str, new_name: str) -> "Tensor":
        """Renames tensor with name ``name`` to ``new_name``

        Args:
            name (str): Name of tensor to be renamed.
            new_name (str): New name of tensor.

        Returns:
            Tensor: Renamed tensor.

        Raises:
            TensorDoesNotExistError: If tensor of name ``name`` does not exist in the dataset.
            TensorAlreadyExistsError: Duplicate tensors are not allowed.
            TensorGroupAlreadyExistsError: Duplicate tensor groups are not allowed.
            InvalidTensorNameError: If ``new_name`` is in dataset attributes.
            RenameError: If ``new_name`` points to a group different from ``name``.
        """
        deeplake_reporter.feature_report(
            feature_name="rename_tensor",
            parameters={"name": name, "new_name": new_name},
        )

        auto_checkout(self)

        if name not in self._tensors():
            raise TensorDoesNotExistError(name)

        name = filter_name(name, self.group_index)
        new_name = filter_name(new_name, self.group_index)

        if posixpath.split(name)[0] != posixpath.split(new_name)[0]:
            raise RenameError("New name of tensor cannot point to a different group")

        if new_name in self.version_state["tensor_names"]:
            raise TensorAlreadyExistsError(new_name)

        if new_name in self._groups:
            raise TensorGroupAlreadyExistsError(new_name)

        new_tensor_name = posixpath.split(new_name)[1]
        if not new_tensor_name or new_tensor_name in dir(self):
            raise InvalidTensorNameError(new_name)

        tensor = self.root._rename_tensor(name, new_name)

        self.storage.maybe_flush()
        return tensor

    def rename_group(self, name: str, new_name: str) -> None:
        """Renames group with name ``name`` to ``new_name``

        Args:
            name (str): Name of group to be renamed.
            new_name (str): New name of group.

        Raises:
            TensorGroupDoesNotExistError: If tensor group of name ``name`` does not exist in the dataset.
            TensorAlreadyExistsError: Duplicate tensors are not allowed.
            TensorGroupAlreadyExistsError: Duplicate tensor groups are not allowed.
            InvalidTensorGroupNameError: If ``name`` is in dataset attributes.
            RenameError: If ``new_name`` points to a group different from ``name``.
        """

        deeplake_reporter.feature_report(
            feature_name="rename_group",
            parameters={"name": name, "new_name": new_name},
        )
        auto_checkout(self)

        name = filter_name(name, self.group_index)
        new_name = filter_name(new_name, self.group_index)

        if name not in self._groups:
            raise TensorGroupDoesNotExistError(name)

        if posixpath.split(name)[0] != posixpath.split(new_name)[0]:
            raise RenameError("Names does not match.")

        if new_name in self.version_state["tensor_names"]:
            raise TensorAlreadyExistsError(new_name)

        if new_name in self._groups:
            raise TensorGroupAlreadyExistsError(new_name)

        new_tensor_name = posixpath.split(new_name)[1]
        if not new_tensor_name or new_tensor_name in dir(self):
            raise InvalidTensorGroupNameError(new_name)

        meta = self.meta
        meta.rename_group(name, new_name)

        root = self.root
        for tensor in filter(
            lambda x: x.startswith(name),
            map(lambda y: y.meta.name or y.key, self.tensors.values()),
        ):
            root._rename_tensor(
                tensor,
                posixpath.join(new_name, relpath(tensor, name)),
            )

        self.storage.maybe_flush()

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except TensorDoesNotExistError as ke:
            raise AttributeError(
                f"'{self.__class__}' object has no attribute '{key}'"
            ) from ke

    def __setattr__(self, name: str, value):
        try:
            # Dataset is not fully loaded if meta is not in version_state
            if "meta" in self.version_state:
                return self.__setitem__(name, value)
            raise TensorDoesNotExistError(name)
        except TensorDoesNotExistError:
            if isinstance(value, (np.ndarray, np.generic)):
                raise TypeError(
                    "Setting tensor attributes directly is not supported. To add a tensor, use the `create_tensor` method."
                    + "To add data to a tensor, use the `append` and `extend` methods."
                )
        return super().__setattr__(name, value)

    def __iter__(self):
        dataset_read(self)
        for i in range(self.__len__(warn=False)):
            yield self.__getitem__(
                i, is_iteration=not isinstance(self.index.values[0], list)
            )

    def _get_commit_id_for_address(self, address, version_state):
        if address in version_state["branch_commit_map"]:
            branch = address
            commit_id = version_state["branch_commit_map"][branch]
        elif address in version_state["commit_node_map"]:
            commit_id = address
        else:
            raise CheckoutError(
                f"Address {address} not found. Ensure the commit id / branch name is correct."
            )
        return commit_id

    def _load_version_info(self, address=None):
        """Loads data from version_control_file otherwise assume it doesn't exist and load all empty"""
        if self.version_state:
            return

        if address is None:
            address = "main"

        version_state = {}
        try:
            try:
                version_info = load_version_info(self.storage)
            except Exception as e:
                version_info = rebuild_version_info(self.storage)
                if version_info is None:
                    raise e
            version_state["branch_commit_map"] = version_info["branch_commit_map"]
            version_state["commit_node_map"] = version_info["commit_node_map"]

            commit_id = self._get_commit_id_for_address(address, version_state)

            version_state["commit_id"] = commit_id
            version_state["commit_node"] = version_state["commit_node_map"][commit_id]
            version_state["branch"] = version_state["commit_node"].branch
        except Exception as e:
            if isinstance(e, CheckoutError):
                raise e from None
            if address != "main":
                raise CheckoutError(
                    f"Address {address} not found. Ensure the commit id / branch name is correct."
                )
            branch = "main"
            version_state["branch"] = branch
            version_state["branch_commit_map"] = {}
            version_state["commit_node_map"] = {}
            # used to identify that this is the first commit so its data will not be in similar directory structure to the rest
            commit_id = FIRST_COMMIT_ID
            commit_node = CommitNode(branch, commit_id)
            version_state["commit_id"] = commit_id
            version_state["commit_node"] = commit_node
            version_state["branch_commit_map"][branch] = commit_id
            version_state["commit_node_map"][commit_id] = commit_node
        # keeps track of the full unindexed tensors
        version_state["full_tensors"] = {}
        version_state["tensor_names"] = {}
        self.__dict__["version_state"] = version_state

    def _load_link_creds(self):
        if self.link_creds is not None:
            return

        link_creds_key = get_dataset_linked_creds_key()
        try:
            data_bytes = self.storage[link_creds_key]
        except KeyError:
            data_bytes = None
        if data_bytes is None:
            link_creds = LinkCreds()
        else:
            link_creds = LinkCreds.frombuffer(data_bytes)
        self.link_creds = link_creds

    def regenerate_vdb_indexes(self):
        tensors = self.tensors

        for _, tensor in tensors.items():
            is_embedding = tensor.htype == "embedding"
            has_vdb_indexes = hasattr(tensor.meta, "vdb_indexes")
            try:
                vdb_index_ids_present = len(tensor.meta.vdb_indexes) > 0
            except AttributeError:
                vdb_index_ids_present = False

            if is_embedding and has_vdb_indexes and vdb_index_ids_present:
                tensor._regenerate_vdb_indexes()

    def _lock(self, err=False, verbose=True):
        if not self.is_head_node or not self._locking_enabled:
            return True
        storage = self.base_storage
        if storage.read_only and not self._locked_out:
            if err:
                raise ReadOnlyModeError()
            return False

        if isinstance(storage, tuple(_LOCKABLE_STORAGES)) and (
            not self.read_only or self._locked_out
        ):
            if not deeplake.constants.LOCKS_ENABLED:
                return True
            try:
                # temporarily disable read only on base storage, to try to acquire lock, if exception, it will be again made readonly
                storage.disable_readonly()
                lock_dataset(
                    self,
                    lock_lost_callback=self._lock_lost_handler,
                )
            except LockedException as e:
                self._set_read_only(True, False)
                self.__dict__["_locked_out"] = True
                if err:
                    raise e
                if verbose and self.verbose:
                    always_warn(
                        "Checking out dataset in read only mode as another machine has locked this version for writing."
                    )
                return False
        return True

    def _unlock(self):
        unlock_dataset(self)

    def __del__(self):
        if self._view_base:
            view_id = self._view_id
            try:
                del self._view_base._commit_hooks[view_id]
            except KeyError:
                pass

            try:
                del self._view_base._checkout_hooks[view_id]
            except KeyError:
                pass

            try:
                del self._view_base._update_hooks[view_id]
            except KeyError:
                pass
        try:
            self._unlock()
        except Exception:  # python shutting down
            pass

    @spinner
    @invalid_view_op
    def commit(self, message: Optional[str] = None, allow_empty=False) -> str:
        """Stores a snapshot of the current state of the dataset.

        Args:
            message (str, Optional): Used to describe the commit.
            allow_empty (bool): If ``True``, commit even if there are no changes.

        Returns:
            str: the commit id of the saved commit that can be used to access the snapshot.

        Raises:
            Exception: If dataset is a filtered view.
            EmptyCommitError: if there are no changes and user does not forced to commit unchanged data.

        Note:
            - Commiting from a non-head node in any branch, will lead to an automatic checkout to a new branch.
            - This same behaviour will happen if new samples are added or existing samples are updated from a non-head node.
        """

        # do not store commit message
        deeplake_reporter.feature_report(
            feature_name="commit", parameters={"allow_empty": allow_empty}
        )

        if not allow_empty and not self.has_head_changes:
            raise EmptyCommitError(
                "There are no changes, commit is not done. Try again with allow_empty=True."
            )

        return self._commit(message, None, False)

    @spinner
    @invalid_view_op
    @suppress_iteration_warning
    def merge(
        self,
        target_id: str,
        conflict_resolution: Optional[str] = None,
        delete_removed_tensors: bool = False,
        force: bool = False,
    ):
        """Merges the target_id into the current dataset.

        Args:
            target_id (str): The commit_id or branch to merge.
            conflict_resolution (str, Optional):
                - The strategy to use to resolve merge conflicts.
                - Conflicts are scenarios where both the current dataset and the target id have made changes to the same sample/s since their common ancestor.
                - Must be one of the following
                    - None - this is the default value, will raise an exception if there are conflicts.
                    - "ours" - during conflicts, values from the current dataset will be used.
                    - "theirs" - during conflicts, values from target id will be used.
            delete_removed_tensors (bool): If ``True``, deleted tensors will be deleted from the dataset.
            force (bool):
                - Forces merge.
                - ``force=True`` will have these effects in the following cases of merge conflicts:
                    - If tensor is renamed on target but is missing from HEAD, renamed tensor will be registered as a new tensor on current branch.
                    - If tensor is renamed on both target and current branch, tensor on target will be registered as a new tensor on current branch.
                    - If tensor is renamed on target and a new tensor of the new name was created on the current branch, they will be merged.

        Raises:
            Exception: if dataset is a filtered view.
            ValueError: if the conflict resolution strategy is not one of the None, "ours", or "theirs".
        """

        deeplake_reporter.feature_report(
            feature_name="merge",
            parameters={
                "target_id": target_id,
                "conflict_resolution": conflict_resolution,
                "delete_removed_tensors": delete_removed_tensors,
                "force": force,
            },
        )

        if self._is_filtered_view:
            raise Exception(
                "Cannot perform version control operations on a filtered dataset view."
            )

        if conflict_resolution not in [None, "ours", "theirs"]:
            raise ValueError(
                f"conflict_resolution must be one of None, 'ours', or 'theirs'. Got {conflict_resolution}"
            )

        try_flushing(self)

        target_commit = target_id
        try:
            target_commit = self.version_state["branch_commit_map"][target_id]
        except KeyError:
            pass
        if (
            isinstance(self.base_storage, tuple(_LOCKABLE_STORAGES))
            and not deeplake.constants.LOCKS_ENABLED
        ):
            lock_dataset(self, version=target_commit)
            locked = True
        else:
            locked = False
        self._initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False
        try:
            merge(self, target_id, conflict_resolution, delete_removed_tensors, force)
        finally:
            if locked:
                unlock_dataset(self, version=target_commit)
            self.storage.autoflush = self._initial_autoflush.pop()
            self.storage.maybe_flush()

    def _commit(
        self,
        message: Optional[str] = None,
        hash: Optional[str] = None,
        flush_version_control_info: bool = True,
        *,
        is_checkpoint: bool = False,
        total_samples_processed: int = 0,
    ) -> str:
        if self._is_filtered_view:
            raise Exception(
                "Cannot perform version control operations on a filtered dataset view."
            )

        try_flushing(self)

        self._initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False
        try:
            self._unlock()
            commit(
                self,
                message,
                hash,
                flush_version_control_info,
                is_checkpoint=is_checkpoint,
                total_samples_processed=total_samples_processed,
            )
            if not flush_version_control_info:
                self.__dict__["_vc_info_updated"] = True
            self._lock()
        finally:
            self.storage.autoflush = self._initial_autoflush.pop()
        self._info = None
        self._ds_diff = None
        [f() for f in list(self._commit_hooks.values())]
        self.maybe_flush()
        return self.commit_id  # type: ignore

    @invalid_view_op
    def checkout(
        self, address: str, create: bool = False, reset: bool = False
    ) -> Optional[str]:
        """Checks out to a specific commit_id or branch. If ``create = True``, creates a new branch with name ``address``.

        Args:
            address (str): The commit_id or branch to checkout to.
            create (bool): If ``True``, creates a new branch with name as address.
            reset (bool): If checkout fails due to a corrupted HEAD state of the branch, setting ``reset=True`` will
                          reset HEAD changes and attempt the checkout again.

        Returns:
            Optional[str]: The commit_id of the dataset after checkout.

        Raises:
            CheckoutError: If ``address`` could not be found.
            ReadOnlyModeError: If branch creation or reset is attempted in read-only mode.
            DatasetCorruptError: If checkout failed due to dataset corruption and ``reset`` is not ``True``.
            Exception: If the dataset is a filtered view.

        Examples:

            >>> ds = deeplake.empty("../test/test_ds")
            >>> ds.create_tensor("abc")
            Tensor(key='abc')
            >>> ds.abc.append([1, 2, 3])
            >>> first_commit = ds.commit()
            >>> ds.checkout("alt", create=True)
            'firstdbf9474d461a19e9333c2fd19b46115348f'
            >>> ds.abc.append([4, 5, 6])
            >>> ds.abc.numpy()
            array([[1, 2, 3],
                   [4, 5, 6]])
            >>> ds.checkout(first_commit)
            'firstdbf9474d461a19e9333c2fd19b46115348f'
            >>> ds.abc.numpy()
            array([[1, 2, 3]])

        Note:
            Checkout from a head node in any branch that contains uncommitted data will lead to an automatic commit before the checkout.
        """

        # do not store address
        deeplake_reporter.feature_report(
            feature_name="checkout",
            parameters={"create": create, "reset": reset},
        )

        try:
            ret = self._checkout(address, create, None, False)
            integrity_check(self)
            return ret
        except (ReadOnlyModeError, CheckoutError) as e:
            raise e from None
        except Exception as e:
            if create:
                raise e
            if not reset:
                if isinstance(e, DatasetCorruptError):
                    raise DatasetCorruptError(
                        message=e.message,
                        action="Try using `reset=True` to reset HEAD changes and load the previous commit.",
                        cause=e.__cause__,
                    )
                raise DatasetCorruptError(
                    "Exception occured (see Traceback). The branch you are checking out to maybe corrupted."
                    "Try using `reset=True` to reset HEAD changes and load the previous commit."
                    "This will delete all uncommitted changes on the branch you are trying to load."
                ) from e
            if self.read_only:
                raise ReadOnlyModeError("Cannot reset HEAD in read-only mode.")
            return reset_and_checkout(self, address, e)

    def _checkout(
        self,
        address: str,
        create: bool = False,
        hash: Optional[str] = None,
        verbose: bool = True,
        flush_version_control_info: bool = False,
    ) -> Optional[str]:
        if self._is_filtered_view:
            raise Exception(
                "Cannot perform version control operations on a filtered dataset view."
            )
        read_only = self._read_only
        if read_only and create:
            raise ReadOnlyModeError()
        try_flushing(self)
        self._initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False
        try:
            self._unlock()
            checkout(self, address, create, hash, flush_version_control_info)
            if not flush_version_control_info and create:
                self.__dict__["_vc_info_updated"] = True
        finally:
            self._set_read_only(read_only, err=True)
            self.storage.autoflush = self._initial_autoflush.pop()
        self._info = None
        self._ds_diff = None

        [f() for f in list(self._checkout_hooks.values())]

        commit_node = self.version_state["commit_node"]
        if self.verbose:
            warn_node_checkout(commit_node, create)
        if create:
            self.maybe_flush()
        return self.commit_id

    @invalid_view_op
    def delete_branch(self, name: str) -> None:
        """
        Deletes the branch and cleans up any unneeded data.
        Branches can only be deleted if there are no sub-branches and if it has never been merged into another branch.

        Args:
            name (str): The branch to delete.

        Raises:
            CommitError: If ``branch`` could not be found.
            ReadOnlyModeError: If branch deletion is attempted in read-only mode.
            Exception: If you have the given branch currently checked out.

        Examples:

            >>> ds = deeplake.empty("../test/test_ds")
            >>> ds.create_tensor("abc")
            Tensor(key='abc')
            >>> ds.abc.append([1, 2, 3])
            >>> first_commit = ds.commit()
            >>> ds.checkout("alt", create=True)
            'firstdbf9474d461a19e9333c2fd19b46115348f'
            >>> ds.abc.append([4, 5, 6])
            >>> ds.abc.numpy()
            array([[1, 2, 3],
                   [4, 5, 6]])
            >>> ds.checkout(first_commit)
            'firstdbf9474d461a19e9333c2fd19b46115348f'
            >>> ds.delete_branch("alt")
        """
        deeplake_reporter.feature_report(
            feature_name="branch_delete",
            parameters={},
        )

        self._delete_branch(name)
        integrity_check(self)

    def _delete_branch(self, name: str) -> None:
        if self._is_filtered_view:
            raise Exception(
                "Cannot perform version control operations on a filtered dataset view."
            )
        read_only = self._read_only
        if read_only:
            raise ReadOnlyModeError()
        try_flushing(self)
        self._initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False
        try:
            self._unlock()
            delete_branch(self, name)
        finally:
            self._set_read_only(read_only, err=True)
            self.storage.autoflush = self._initial_autoflush.pop()

    @invalid_view_op
    def _squash_main(self) -> None:
        """
        DEPRECATED: This method is deprecated and will be removed in a future release.

        Squashes all commits in current branch into one commit.
        NOTE: This cannot be run if there are any branches besides ``main``

        Raises:
            ReadOnlyModeError: If branch deletion is attempted in read-only mode.
            VersionControlError: If the branch cannot be squashed.
            Exception: If the dataset is filtered view.
        """
        if self._is_filtered_view:
            raise Exception(
                "Cannot perform version control operations on a filtered dataset view."
            )
        read_only = self._read_only
        if read_only:
            raise ReadOnlyModeError()

        try_flushing(self)

        self._initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False
        try:
            self._unlock()
            _squash_main(self)
        finally:
            self._set_read_only(read_only, err=True)
            self.libdeeplake_dataset = None
            self.storage.autoflush = self._initial_autoflush.pop()

    def log(self):
        """Displays the details of all the past commits."""

        deeplake_reporter.feature_report(feature_name="log", parameters={})

        commit_node = self.version_state["commit_node"]
        print("---------------\nDeep Lake Version Log\n---------------\n")
        print(f"Current Branch: {self.version_state['branch']}")
        if self.has_head_changes:
            print("** There are uncommitted changes on this branch.")
        print()
        while commit_node:
            if not commit_node.is_head_node:
                print(f"{commit_node}\n")
            commit_node = commit_node.parent

    def diff(
        self, id_1: Optional[str] = None, id_2: Optional[str] = None, as_dict=False
    ) -> Optional[Dict]:
        """Returns/displays the differences between commits/branches.

        For each tensor this contains information about the sample indexes that were added/modified as well as whether the tensor was created.

        Args:
            id_1 (str, Optional): The first commit_id or branch name.
            id_2 (str, Optional): The second commit_id or branch name.
            as_dict (bool, Optional): If ``True``, returns the diff as lists of commit wise dictionaries.

        Returns:
            Optional[Dict]

        Raises:
            ValueError: If ``id_1`` is None and ``id_2`` is not None.

        Note:
            - If both ``id_1`` and ``id_2`` are None, the differences between the current state and the previous commit will be calculated. If you're at the head of the branch, this will show the uncommitted changes, if any.
            - If only ``id_1`` is provided, the differences between the current state and id_1 will be calculated. If you're at the head of the branch, this will take into account the uncommitted changes, if any.
            - If only ``id_2`` is provided, a ValueError will be raised.
            - If both ``id_1`` and ``id_2`` are provided, the differences between ``id_1`` and ``id_2`` will be calculated.

        Note:
            A dictionary of the differences between the commits/branches is returned if ``as_dict`` is ``True``.
            The dictionary will always have 2 keys, "dataset" and "tensors". The values corresponding to these keys are detailed below:

                - If ``id_1`` and ``id_2`` are None, both the keys will have a single list as their value. This list will contain a dictionary describing changes compared to the previous commit.
                - If only ``id_1`` is provided, both keys will have a tuple of 2 lists as their value. The lists will contain dictionaries describing commitwise differences between commits. The 2 lists will range from current state and ``id_1`` to most recent common ancestor the commits respectively.
                - If only ``id_2`` is provided, a ValueError will be raised.
                - If both ``id_1`` and ``id_2`` are provided, both keys will have a tuple of 2 lists as their value. The lists will contain dictionaries describing commitwise differences between commits. The 2 lists will range from ``id_1`` and ``id_2`` to most recent common ancestor the commits respectively.

            ``None`` is returned if ``as_dict`` is ``False``.
        """

        deeplake_reporter.feature_report(
            feature_name="diff", parameters={"as_dict": str(as_dict)}
        )

        version_state, storage = self.version_state, self.storage
        res = get_changes_and_messages(version_state, storage, id_1, id_2)
        if as_dict:
            dataset_changes_1 = res[0]
            dataset_changes_2 = res[1]
            tensor_changes_1 = res[2]
            tensor_changes_2 = res[3]
            changes = {}
            if id_1 is None and id_2 is None:
                changes["dataset"] = dataset_changes_1
                changes["tensor"] = tensor_changes_1
                return changes
            changes["dataset"] = dataset_changes_1, dataset_changes_2
            changes["tensor"] = tensor_changes_1, tensor_changes_2
            return changes
        all_changes = get_all_changes_string(*res)
        print(all_changes)
        return None

    def _populate_meta(self, address: Optional[str] = None, verbose=True):
        """Populates the meta information for the dataset."""
        if address is None:
            commit_id = self._get_commit_id_for_address("main", self.version_state)
        else:
            commit_id = self._get_commit_id_for_address(address, self.version_state)

        if dataset_exists(self.storage, commit_id):
            load_meta(self)

        elif not self.storage.empty():
            # dataset does not exist, but the path was not empty
            raise PathNotEmptyException

        else:
            if self.read_only:
                # cannot create a new dataset when in read_only mode.
                raise CouldNotCreateNewDatasetException(self.path)
            meta = DatasetMeta()
            key = get_dataset_meta_key(self.version_state["commit_id"])
            self.version_state["meta"] = meta
            self.storage.register_deeplake_object(key, meta)
            self._register_dataset()
            self.flush()

    def _register_dataset(self):
        if not self.__dict__["org_id"]:
            self.org_id = os.environ.get(ENV_HUB_DEV_USERNAME)

    def _send_query_progress(self, *args, **kwargs):
        """overridden in DeepLakeCloudDataset"""

    def _send_compute_progress(self, *args, **kwargs):
        """overridden in DeepLakeCloudDataset"""

    def _send_pytorch_progress(self, *args, **kwargs):
        """overridden in DeepLakeCloudDataset"""

    def _send_filter_progress(self, *args, **kwargs):
        """overridden in DeepLakeCloudDataset"""

    def _send_commit_event(self, *args, **kwargs):
        """overridden in DeepLakeCloudDataset"""

    def _send_dataset_creation_event(self, *args, **kwargs):
        """overridden in DeepLakeCloudDataset"""

    def _send_branch_creation_event(self, *args, **kwargs):
        """overridden in DeepLakeCloudDataset"""

    def _send_branch_deletion_event(self, *args, **kwargs):
        """overridden in DeepLakeCloudDataset"""

    def _first_load_init(self, verbose=True):
        """overridden in DeepLakeCloudDataset"""

    @property
    def read_only(self):
        """Returns True if dataset is in read-only mode and False otherwise."""
        return self._read_only

    @property
    def is_head_node(self):
        """Returns True if the current commit is the head node of the branch and False otherwise."""
        commit_node = self.version_state["commit_node"]
        return not commit_node.children

    @property
    def has_head_changes(self):
        """Returns True if currently at head node and uncommitted changes are present."""
        return self.is_head_node and current_commit_has_change(
            self.version_state, self.storage
        )

    def _set_read_only(self, value: bool, err: bool):
        storage = self.storage
        self.__dict__["_read_only"] = value

        if value:
            storage.enable_readonly()
            if isinstance(storage, LRUCache) and storage.next_storage is not None:
                storage.next_storage.enable_readonly()
            self._unlock()
        else:
            try:
                locked = self._lock(err=err)
                if locked:
                    self.storage.disable_readonly()
                    if (
                        isinstance(storage, LRUCache)
                        and storage.next_storage is not None
                    ):
                        storage.next_storage.disable_readonly()
                else:
                    self.__dict__["_read_only"] = True
            except LockedException as e:
                self.__dict__["_read_only"] = True
                if err:
                    raise e

    @read_only.setter
    @invalid_view_op
    def read_only(self, value: bool):
        self._set_read_only(value, True)

    @property
    def allow_delete(self) -> bool:
        """Returns True if dataset can be deleted from storage. Whether it can be deleted or not is stored in the database_meta.json and can be changed with `allow_delete = True|False`"""
        return self.meta.allow_delete

    @allow_delete.setter
    def allow_delete(self, value: bool):
        self.meta.allow_delete = value
        self.flush()

    def pytorch(
        self,
        transform: Optional[Callable] = None,
        tensors: Optional[Sequence[str]] = None,
        num_workers: int = 1,
        batch_size: int = 1,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        shuffle: bool = False,
        buffer_size: int = 2048,
        use_local_cache: bool = False,
        progressbar: bool = False,
        return_index: bool = True,
        pad_tensors: bool = False,
        transform_kwargs: Optional[Dict[str, Any]] = None,
        decode_method: Optional[Dict[str, str]] = None,
        cache_size: int = 32 * MB,
        *args,
        **kwargs,
    ):
        """Creates a PyTorch Dataloader from the Deep Lake dataset. During iteration, the data from all tensors will be streamed on-the-fly from the storage location.
        Understanding the parameters below is critical for achieving fast streaming for your use-case

        Args:
            *args: Additional args to be passed to torch_dataset
            **kwargs: Additional kwargs to be passed to torch_dataset
            transform (Callable, Optional): Transformation function to be applied to each sample.
            tensors (List, Optional): List of tensors to load. If ``None``, all tensors are loaded. Defaults to ``None``.
                For datasets with many tensors, its extremely important to stream only the data that is needed for training the model, in order to avoid bottlenecks associated with streaming unused data.
                For example, if you have a dataset that has ``image``, ``label``, and ``metadata`` tensors, if ``tensors=["image", "label"]``, the Data Loader will only stream the ``image`` and ``label`` tensors.
            num_workers (int): The number of workers to use for fetching data in parallel.
            batch_size (int): Number of samples per batch to load. Default value is 1.
            drop_last (bool): Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
                if ``False`` and the size of dataset is not divisible by the batch size, then the last batch will be smaller. Default value is ``False``.
                Read torch.utils.data.DataLoader docs for more details.
            collate_fn (Callable, Optional): merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
                Read torch.utils.data.DataLoader docs for more details.
            pin_memory (bool): If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them. Default value is ``False``.
                Read torch.utils.data.DataLoader docs for more details.
            shuffle (bool): If ``True``, the data loader will shuffle the data indices. Default value is False. Details about how Deep Lake shuffles data can be found at `Shuffling in ds.pytorch() <https://docs.activeloop.ai/how-it-works/shuffling-in-ds.pytorch>`_
            buffer_size (int): The size of the buffer used to shuffle the data in MBs. Defaults to 2048 MB. Increasing the buffer_size will increase the extent of shuffling.
            use_local_cache (bool): If ``True``, the data loader will use a local cache to store data. The default cache location is ~/.activeloop/cache, but it can be changed by setting the ``LOCAL_CACHE_PREFIX`` environment variable. This is useful when the dataset can fit on the machine and we don't want to fetch the data multiple times for each iteration. Default value is ``False``
            progressbar (bool): If ``True``, tqdm will be wrapped around the returned dataloader. Default value is True.
            return_index (bool): If ``True``, the returned dataloader will have a key "index" that contains the index of the sample(s) in the original dataset. Default value is True.
            pad_tensors (bool): If ``True``, shorter tensors will be padded to the length of the longest tensor. Default value is False.
            transform_kwargs (optional, Dict[str, Any]): Additional kwargs to be passed to ``transform``.
            decode_method (Dict[str, str], Optional): The method for decoding the Deep Lake tensor data, the result of which is passed to the transform. Decoding occurs outside of the transform so that it can be performed in parallel and as rapidly as possible as per Deep Lake optimizations.

                - Supported decode methods are:
                    :'numpy': Default behaviour. Returns samples as numpy arrays, the same as ds.tensor[i].numpy()
                    :'tobytes': Returns raw bytes of the samples the same as ds.tensor[i].tobytes()
                    :'data': Returns a dictionary with keys,values depending on htype, the same as ds.tensor[i].data()
                    :'pil': Returns samples as PIL images. Especially useful when transformation use torchvision transforms, that
                            require PIL images as input. Only supported for tensors with ``sample_compression='jpeg'`` or ``'png'``.

            cache_size (int): The size of the cache per tensor in MBs. Defaults to max(maximum chunk size of tensor, 32 MB).

        ..
            # noqa: DAR101

        Returns:
            A torch.utils.data.DataLoader object.

        Raises:
            EmptyTensorError: If one or more tensors being passed to pytorch are empty.

        Note:
            Pytorch does not support uint16, uint32, uint64 dtypes. These are implicitly type casted to int32, int64 and int64 respectively.
            This spins up it's own workers to fetch data.
        """
        from deeplake.integrations import dataset_to_pytorch as to_pytorch

        deeplake_reporter.feature_report(
            feature_name="pytorch",
            parameters={
                "tensors": tensors,
                "num_workers": num_workers,
                "batch_size": batch_size,
                "drop_last": drop_last,
                "pin_memory": pin_memory,
                "shuffle": shuffle,
                "buffer_size": buffer_size,
                "use_local_cache": use_local_cache,
                "progressbar": progressbar,
                "return_index": return_index,
                "pad_tensors": pad_tensors,
                "decode_method": decode_method,
            },
        )

        if transform and transform_kwargs:
            transform = partial(transform, **transform_kwargs)

        dataloader = to_pytorch(
            self,
            *args,
            transform=transform,
            tensors=tensors,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            shuffle=shuffle,
            buffer_size=buffer_size,
            use_local_cache=use_local_cache,
            return_index=return_index,
            pad_tensors=pad_tensors,
            decode_method=decode_method,
            cache_size=cache_size,
            **kwargs,
        )

        if progressbar:
            dataloader = tqdm(
                dataloader, desc=self.path, total=self.__len__(warn=False) // batch_size
            )
        dataset_read(self)
        return dataloader

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

        """
        from deeplake.enterprise.dataloader import dataloader

        deeplake_reporter.feature_report(feature_name="dataloader", parameters={})

        return dataloader(self, ignore_errors=ignore_errors, verbose=verbose)

    def filter(
        self,
        function: Union[Callable, str],
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,
        save_result: bool = False,
        result_path: Optional[str] = None,
        result_ds_args: Optional[dict] = None,
    ):
        """Filters the dataset in accordance of filter function ``f(x: sample) -> bool``

        Args:
            function (Callable, str): Filter function that takes sample as argument and returns ``True`` / ``False``
                if sample should be included in result. Also supports simplified expression evaluations.
                See :class:`deeplake.core.query.query.DatasetQuery` for more details.
            num_workers (int): Level of parallelization of filter evaluations.
                0 indicates in-place for-loop evaluation, multiprocessing is used otherwise.
            scheduler (str): Scheduler to use for multiprocessing evaluation.
                "threaded" is default.
            progressbar (bool): Display progress bar while filtering. ``True`` is default.
            save_result (bool): If ``True``, result of the filter will be saved to a dataset asynchronously.
            result_path (Optional, str): Path to save the filter result. Only applicable if ``save_result`` is True.
            result_ds_args (Optional, dict): Additional args for result dataset. Only applicable if ``save_result`` is True.

        Returns:
            View of Dataset with elements that satisfy filter function.


        Example:
            Return dataset view where all the samples have label equals to 2:

            >>> dataset.filter(lambda sample: sample.labels.numpy() == 2)


            Append one dataset onto another (only works if their structure is identical):

            >>> @deeplake.compute
            >>> def dataset_append(sample_in, sample_out):
            >>>
            >>>     sample_out.append(sample_in.tensors)
            >>>
            >>>     return sample_out
            >>>
            >>>
            >>> dataset_append().eval(
            >>>                 ds_in,
            >>>                 ds_out,
            >>>                 num_workers = 2
            >>>            )
        """
        from deeplake.core.query import filter_dataset, query_dataset

        deeplake_reporter.feature_report(
            feature_name="filter",
            parameters={
                "num_workers": num_workers,
                "scheduler": scheduler,
                "progressbar": progressbar,
                "save_result": save_result,
            },
        )

        fn = query_dataset if isinstance(function, str) else filter_dataset
        ret = fn(
            self,
            function,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
            save_result=save_result,
            result_path=result_path,
            result_ds_args=result_ds_args,
        )
        dataset_read(self)
        return ret

    def query(
        self,
        query_string: str,
        runtime: Optional[Dict] = None,
        return_data: bool = False,
    ):
        """Returns a sliced :class:`~deeplake.core.dataset.Dataset` with given query results.

        It allows to run SQL like queries on dataset and extract results. See supported keywords and the Tensor Query Language documentation
        :ref:`here <tql>`.


        Args:
            query_string (str): An SQL string adjusted with new functionalities to run on the given :class:`~deeplake.core.dataset.Dataset` object
            runtime (Optional[Dict]): Runtime parameters for query execution. Supported keys: {"tensor_db": True or False}.
            return_data (bool): Defaults to ``False``. Whether to return raw data along with the view.

        Raises:
            ValueError: if ``return_data`` is True and runtime is not {"tensor_db": true}

        Returns:
            Dataset: A :class:`~deeplake.core.dataset.Dataset` object.

        Examples:

            Query from dataset all the samples with lables other than ``5``

            >>> import deeplake
            >>> ds = deeplake.load('hub://activeloop/fashion-mnist-train')
            >>> query_ds = ds.query("select * where labels != 5")

            Query from dataset first appeard ``1000`` samples where the ``categories`` is ``car`` and ``1000`` samples where the ``categories`` is ``motorcycle``

            >>> ds_train = deeplake.load('hub://activeloop/coco-train')
            >>> query_ds_train = ds_train.query("(select * where contains(categories, 'car') limit 1000) union (select * where contains(categories, 'motorcycle') limit 1000)")

        """

        deeplake_reporter.feature_report(
            feature_name="query",
            parameters={
                "query_string": query_string[0:100],
                "runtime": runtime,
            },
        )

        runtime = parse_runtime_parameters(self.path, runtime)
        if runtime["tensor_db"]:
            client = DeepLakeBackendClient(token=self._token)
            org_id, ds_name = self.path[6:].split("/")
            response = client.remote_query(org_id, ds_name, query_string)
            indices = response["indices"]
            view = self[indices]

            if return_data:
                data = response["data"]
                return view, data

            return view

        if return_data:
            raise ValueError(
                "`return_data` is only applicable when running queries using the Managed Tensor Database. Please specify `runtime = {'tensor_db': True}`"
            )

        from deeplake.enterprise import query

        result = query(self, query_string)

        if len(query_string) > QUERY_MESSAGE_MAX_SIZE:
            message = query_string[: QUERY_MESSAGE_MAX_SIZE - 3] + "..."
        else:
            message = query_string
        result._query_string = message

        return result

    def sample_by(
        self,
        weights: Union[str, list, tuple],
        replace: Optional[bool] = True,
        size: Optional[int] = None,
    ):
        """Returns a sliced :class:`~deeplake.core.dataset.Dataset` with given weighted sampler applied.

        Args:
            weights: (Union[str, list, tuple]): If it's string then tql will be run to calculate the weights based on the expression. list and tuple will be treated as the list of the weights per sample.
            replace: Optional[bool] If true the samples can be repeated in the result view. Defaults to ``True``
            size: Optional[int] The length of the result view. Defaults to length of the dataset.


        Returns:
            Dataset: A deeplake.Dataset object.

        Examples:

            Sample the dataset with ``labels == 5`` twice more than ``labels == 6``

            >>> from deeplake.experimental import query
            >>> ds = deeplake.load('hub://activeloop/fashion-mnist-train')
            >>> sampled_ds = ds.sample_by("max_weight(labels == 5: 10, labels == 6: 5)")

            Sample the dataset treating `labels` tensor as weights.

            >>> import deeplake
            >>> ds = deeplake.load('hub://activeloop/fashion-mnist-train')
            >>> sampled_ds = ds.sample_by("max_weight(labels == 5: 10, labels == 6: 5"))

            Sample the dataset with the given weights;

            >>> ds = deeplake.load('hub://activeloop/coco-train')
            >>> weights = list()
            >>> for i in range(len(ds)):
            ...     weights.append(i % 5)
            ...
            >>> sampled_ds = ds.sample_by(weights, replace=False)

        """
        from deeplake.enterprise import sample_by

        deeplake_reporter.feature_report(
            feature_name="sample_by",
            parameters={
                "replace": replace,
                "size": size,
            },
        )

        return sample_by(self, weights, replace, size)

    def _get_total_meta(self):
        """Returns tensor metas all together"""
        return {
            tensor_key: tensor_value.meta
            for tensor_key, tensor_value in self.version_state["full_tensors"].items()
        }

    def _set_derived_attributes(
        self, verbose: bool = True, address: Optional[str] = None
    ):
        """Sets derived attributes during init and unpickling."""
        if self.is_first_load:
            self.storage.autoflush = True
            self._load_version_info(address)
            self._load_link_creds()
            self._set_read_only(
                self._read_only, err=self._read_only_error
            )  # TODO: weird fix for dataset unpickling
            self._populate_meta(
                address, verbose
            )  # TODO: use the same scheme as `load_info`
            if self.index.is_trivial():
                self.index = Index.from_json(self.meta.default_index)
        elif not self._read_only:
            self._lock(verbose=verbose)  # for ref counting

        if not self.is_first_load and not self.group_index:
            self._reload_version_state()

        if not self.is_iteration and not self.index.is_trivial():
            group_index = self.group_index
            group_filter = (
                lambda t: (not group_index or t.key.startswith(group_index + "/"))
                and t.key not in self.meta.hidden_tensors
            )
            group_tensors = filter(
                group_filter, self.version_state["full_tensors"].values()
            )
            max_tensor_length = max(map(len, group_tensors), default=0)
            self.index.validate(max_tensor_length)

    @property
    def info(self):
        """Returns the information about the dataset."""
        if self.group_index:
            raise GroupInfoNotSupportedError
        if self._info is None:
            path = get_dataset_info_key(self.version_state["commit_id"])
            self.__dict__["_info"] = load_info(path, self)  # type: ignore
        return self._info

    @info.setter
    def info(self, value):
        if isinstance(value, dict):
            info = self.info
            info.replace_with(value)
        else:
            raise TypeError("Info must be set with type Dict")

    @property
    def _dataset_diff(self):
        if self._ds_diff is None:
            self.__dict__["_ds_diff"] = load_dataset_diff(self)
        return self._ds_diff

    def tensorflow(
        self,
        tensors: Optional[Sequence[str]] = None,
        tobytes: Union[bool, Sequence[str]] = False,
        fetch_chunks: bool = True,
    ):
        """Converts the dataset into a tensorflow compatible format.

        See https://www.tensorflow.org/api_docs/python/tf/data/Dataset

        Args:
            tensors (List, Optional): Optionally provide a list of tensor names in the ordering that your training script expects. For example, if you have a dataset that has "image" and "label" tensors, if ``tensors=["image", "label"]``, your training script should expect each batch will be provided as a tuple of (image, label).
            tobytes (bool): If ``True``, samples will not be decompressed and their raw bytes will be returned instead of numpy arrays. Can also be a list of tensors, in which case those tensors alone will not be decompressed.
            fetch_chunks: See fetch_chunks argument in deeplake.core.tensor.Tensor.numpy()

        Returns:
            tf.data.Dataset object that can be used for tensorflow training.
        """
        deeplake_reporter.feature_report(
            feature_name="tensorflow",
            parameters={
                "tensors": tensors,
                "tobytes": tobytes,
                "fetch_chunks": fetch_chunks,
            },
        )

        dataset_read(self)
        return dataset_to_tensorflow(
            self, tensors=tensors, tobytes=tobytes, fetch_chunks=fetch_chunks
        )

    @spinner
    def flush(self):
        """
        Writes all the data that has been changed/assigned from the cache layers (if any) to the underlying storage.

        NOTE: The high-level APIs flush the cache automatically and users generally do not have explicitly run the ``flush`` command.
        """
        self._flush_vc_info()
        self.storage.flush()

    def _flush_vc_info(self):
        if self._vc_info_updated:
            save_version_info(self.version_state, self.storage)
            for node in self.version_state["commit_node_map"].values():
                if node._info_updated:
                    save_commit_info(node, self.storage)
            self.__dict__["_vc_info_updated"] = False

    def clear_cache(self):
        """
        Flushes (see :func:`Dataset.flush`) the contents of the cache layers (if any) and then deletes contents of all the layers of it.
        This doesn't delete data from the actual storage.
        This is useful if you have multiple datasets with memory caches open, taking up too much RAM, or when local cache is no longer needed for certain datasets and is taking up storage space.

        NOTE: The high-level APIs clear the cache automatically and users generally do not have explicitly run the ``clear_cache`` command.
        """

        if hasattr(self.storage, "clear_cache"):
            self.storage.clear_cache()

    def size_approx(self):
        """Estimates the size in bytes of the dataset.
        Includes only content, so will generally return an under-estimate.
        """
        tensors = self.version_state["full_tensors"].values()
        chunk_engines = [tensor.chunk_engine for tensor in tensors]
        size = sum(c.num_chunks * c.min_chunk_size for c in chunk_engines)
        for group in self._groups_filtered:
            size += self[group].size_approx()
        return size

    @invalid_view_op
    def rename(self, new_name: str):
        """Renames the dataset to `new_name`.

        Args:
            new_name (str): New name for the dataset.

        Raises:
            RenameError: If this dataset is not a managed dataset
        """

        raise RenameError("Rename is not available for non-managed datasets")

    @invalid_view_op
    def delete(self, large_ok=False):
        """Deletes the entire dataset from the underlying storage and cache layers (if any).
        This is an **IRREVERSIBLE** operation. Data once deleted can not be recovered.

        Args:
            large_ok (bool): Delete datasets larger than 1 GB. Defaults to ``False``.

        Raises:
            DatasetTooLargeToDelete: If the dataset is larger than 1 GB and ``large_ok`` is ``False``.
            DatasetHandlerError: If the dataset is marked as allow_delete=False.
        """

        deeplake_reporter.feature_report(
            feature_name="delete", parameters={"large_ok": large_ok}
        )

        if not self.allow_delete:
            raise DatasetHandlerError(
                "The dataset is marked as allow_delete=false. To delete this dataset, you must first run `allow_delete = True` on the dataset."
            )

        if hasattr(self, "_view_entry"):
            self._view_entry.delete()
            return
        if hasattr(self, "_vds"):
            self._vds.delete(large_ok=large_ok)
            return
        if not large_ok:
            size = self.size_approx()
            if size > deeplake.constants.DELETE_SAFETY_SIZE:
                raise DatasetTooLargeToDelete(self.path)

        self._unlock()

        # Clear out the associated index.
        tensor_dict = self.tensors
        for key, tensor in tensor_dict.items():
            if tensor.htype == "embedding" and hasattr(tensor.meta, "vdb_indexes"):
                indexes = tensor.meta.get_vdb_index_ids()
                for id in indexes:
                    tensor.delete_vdb_index(id)

        self.storage.clear()

    def summary(self, force: bool = False):
        """Prints a summary of the dataset, including the tensor names and their lengths, shapes, htypes, dtypes, compressions, and other relevant information.

        Args:
            force (bool): Dataset views with more than 10000 samples might take several seconds of minutes to summarize. If `force=True`,
                the summary will be printed regardless. An error will be raised otherwise.

        Raises:
            ValueError: If the dataset view might take a long time to summarize and `force=False`
        """

        deeplake_reporter.feature_report(feature_name="summary", parameters={})

        if (
            not self.index.is_trivial()
            and self.max_len >= deeplake.constants.VIEW_SUMMARY_SAFE_LIMIT
            and not force
        ):
            raise ValueError(
                "Dataset views with more than 10000 samples might take a long time to summarize. Use `force=True` to override."
            )

        pretty_print = summary_dataset(self)

        print(self)
        print(pretty_print)

    def __str__(self):
        path_str = ""
        if self.path:
            path_str = f"path='{self.path}', "

        mode_str = ""
        if self.read_only:
            mode_str = f"read_only=True, "

        if not self.allow_delete:
            mode_str = f"allow_delete=False, "

        index_str = f"index={self.index}, "
        if self.index.is_trivial():
            index_str = ""

        group_index_str = (
            f"group_index='{self.group_index}', " if self.group_index else ""
        )

        return f"Dataset({path_str}{mode_str}{index_str}{group_index_str}tensors={self._all_tensors_filtered(include_hidden=False, include_disabled=False)})"

    __repr__ = __str__

    def _get_tensor_from_root(self, name: str) -> Optional[Tensor]:
        """Gets a tensor from the root dataset.
        Acesses storage only for the first call.
        """
        key = self.version_state["tensor_names"].get(name)
        return self.version_state["full_tensors"].get(key)

    def _has_group_in_root(self, name: str) -> bool:
        """Checks if a group exists in the root dataset.
        This is faster than checking ``if group in self._groups:``
        """
        return name in self.version_state["meta"].groups

    @property
    def token(self):
        """Get attached token of the dataset"""
        return self._token or os.environ.get(DEEPLAKE_AUTH_TOKEN)

    @token.setter
    def token(self, new_token: str):
        """Set token to dataset"""
        self._token = new_token

    def set_token(self, new_token: str):
        """Method to set a new token"""
        self._token = new_token

    @property
    def _ungrouped_tensors(self) -> Dict[str, Tensor]:
        """Top level tensors in this group that do not belong to any sub groups"""
        return {
            posixpath.basename(k): self.version_state["full_tensors"][v]
            for k, v in self.version_state["tensor_names"].items()
            if posixpath.dirname(k) == self.group_index
        }

    def _all_tensors_filtered(
        self, include_hidden: bool = True, include_disabled=True
    ) -> List[str]:
        """Names of all tensors belonging to this group, including those within sub groups"""
        hidden_tensors = self.meta.hidden_tensors
        tensor_names = self.version_state["tensor_names"]
        enabled_tensors = self.enabled_tensors
        return [
            relpath(t, self.group_index)
            for t in tensor_names
            if (not self.group_index or t.startswith(self.group_index + "/"))
            and (include_hidden or tensor_names[t] not in hidden_tensors)
            and (include_disabled or enabled_tensors is None or t in enabled_tensors)
        ]

    def _tensors(
        self, include_hidden: bool = True, include_disabled=True
    ) -> Dict[str, Tensor]:
        """All tensors belonging to this group, including those within sub groups. Always returns the sliced tensors."""
        version_state = self.version_state
        index = self.index
        group_index = self.group_index
        all_tensors = self._all_tensors_filtered(include_hidden, include_disabled)
        return {
            t: version_state["full_tensors"][
                version_state["tensor_names"][posixpath.join(group_index, t)]
            ][index]
            for t in all_tensors
        }

    @property
    def tensors(self) -> Dict[str, Tensor]:
        """All tensors belonging to this group, including those within sub groups. Always returns the sliced tensors."""
        return self._tensors(include_hidden=False, include_disabled=False)

    @property
    def branches(self):
        """Lists all the branches of the dataset.

        Returns:
            List of branches.
        """
        return list(self.version_state["branch_commit_map"])

    @property
    def commits(self) -> List[Dict]:
        """Lists all the commits leading to the current dataset state.

        Returns:
            List of dictionaries containing commit information.
        """
        commits = []
        commit_node = self.version_state["commit_node"]
        while commit_node:
            if not commit_node.is_head_node:
                commit_info = {
                    "commit": commit_node.commit_id,
                    "author": commit_node.commit_user_name,
                    "time": str(commit_node.commit_time)[:-7],
                    "message": commit_node.commit_message,
                }
                commits.append(commit_info)
            commit_node = commit_node.parent
        return commits

    def get_commit_details(self, commit_id) -> Dict:
        """Get details of a particular commit.

        Args:
            commit_id (str): commit id of the commit.

        Returns:
            Dict: Dictionary of details with keys - ``commit``, ``author``, ``time``, ``message``.

        Raises:
            KeyError: If given ``commit_id`` is was not found in the dataset.
        """
        commit_node: CommitNode = self.version_state["commit_node_map"].get(commit_id)
        if commit_node is None:
            raise KeyError(f"Commit {commit_id} not found in dataset.")

        time = str(commit_node.commit_time)[:-7] if commit_node.commit_time else None
        return {
            "commit": commit_node.commit_id,
            "author": commit_node.commit_user_name,
            "time": time,
            "message": commit_node.commit_message,
        }

    @property
    def _groups(self) -> List[str]:
        """Names of all groups in the root dataset"""
        return self.meta.groups  # type: ignore

    @property
    def _groups_filtered(self) -> List[str]:
        """Names of all sub groups in this group"""
        groups_filtered = []
        for g in self._groups:
            dirname, basename = posixpath.split(g)
            if dirname == self.group_index:
                groups_filtered.append(basename)
        return groups_filtered

    @property
    def groups(self) -> Dict[str, "Dataset"]:
        """All sub groups in this group"""
        return {g: self[g] for g in self._groups_filtered}

    @property
    def commit_id(self) -> Optional[str]:
        """The lasted committed commit id of the dataset. If there are no commits, this returns ``None``."""
        commit_node = self.version_state["commit_node"]
        if not commit_node.is_head_node:
            return commit_node.commit_id

        parent = commit_node.parent

        if parent is None:
            return None
        else:
            return parent.commit_id

    @property
    def pending_commit_id(self) -> str:
        """The commit_id of the next commit that will be made to the dataset.
        If you're not at the head of the current branch, this will be the same as the commit_id.
        """
        return self.version_state["commit_id"]

    @property
    def branch(self) -> str:
        """The current branch of the dataset"""
        return self.version_state["branch"]

    def _is_root(self) -> bool:
        return not self.group_index

    @property
    def parent(self):
        """Returns the parent of this group. Returns None if this is the root dataset."""
        if self._is_root():
            return None
        autoflush = self.storage.autoflush
        ds = self.__class__(
            storage=self.storage,
            index=self.index,
            group_index=posixpath.dirname(self.group_index),
            read_only=self.read_only,
            public=self.public,
            token=self._token,
            verbose=self.verbose,
            version_state=self.version_state,
            path=self.path,
            link_creds=self.link_creds,
            libdeeplake_dataset=self.libdeeplake_dataset,
            index_params=self.index_params,
        )
        self.storage.autoflush = autoflush
        return ds

    @property
    def root(self):
        """Returns the root dataset of a group."""
        if self._is_root():
            return self
        autoflush = self.storage.autoflush
        ds = self.__class__(
            storage=self.storage,
            index=self.index,
            group_index="",
            read_only=self.read_only,
            public=self.public,
            token=self._token,
            verbose=self.verbose,
            version_state=self.version_state,
            path=self.path,
            link_creds=self.link_creds,
            view_base=self._view_base,
            libdeeplake_dataset=self.libdeeplake_dataset,
            index_params=self.index_params,
        )
        self.storage.autoflush = autoflush
        return ds

    @property
    def no_view_dataset(self):
        """Returns the same dataset without slicing."""
        if self.index is None or self.index.is_trivial():
            return self
        return self.__class__(
            storage=self.storage,
            index=None,
            group_index=self.group_index,
            read_only=self.read_only,
            public=self.public,
            token=self._token,
            verbose=False,
            version_state=self.version_state,
            path=self.path,
            link_creds=self.link_creds,
            pad_tensors=self._pad_tensors,
            enabled_tensors=self.enabled_tensors,
            libdeeplake_dataset=self.libdeeplake_dataset,
            index_params=self.index_params,
        )

    def _create_group(self, name: str) -> "Dataset":
        """Internal method used by `create_group` and `create_tensor`."""
        meta: DatasetMeta = self.version_state["meta"]
        if not name or name in dir(self):
            raise InvalidTensorGroupNameError(name)
        fullname = name
        while name:
            if name in self.version_state["full_tensors"]:
                raise TensorAlreadyExistsError(name)
            meta.add_group(name)
            name, _ = posixpath.split(name)
        return self[fullname]

    def create_group(self, name: str, exist_ok=False) -> "Dataset":
        """Creates a tensor group, which is a collection of tensors that can be reference together.
        Groups are recommended for use-cases with many tensors in order to organize and reference data more easily.

        Args:
            name: The name of the group to create.
            exist_ok: If ``True``, the group is created if it does not exist. If ``False``, an error is raised if the group already exists.
                Defaults to ``False``.

        Returns:
            The created group.

        Raises:
            TensorGroupAlreadyExistsError: If the group already exists and ``exist_ok`` is False.

        Examples:

            >>> ds.create_group("images")
            >>> ds.images.create_tensor("left_camera", htype = "image", "sample_compression" = "jpeg")
            >>> ds.images.create_tensor("center_camera", htype = "image", "sample_compression" = "jpeg")
            >>> ds.images.create_tensor("right_camera", htype = "image", "sample_compression" = "jpeg")

            >>> # Data from a tensors in groups is referenced using:
            >>> ds.images.right_camera[0].numpy()
            >>> # OR
            >>> ds["images/right_camera"].numpy()

            >>> # "/" Notation can also be used to create groups with ``create_tensor``
            >>> ds.create_tensor("images/right_camera", htype = "image", "sample_compression" = "jpeg")
        """

        deeplake_reporter.feature_report(
            feature_name="create_group",
            parameters={
                "name": name,
                "exist_ok": exist_ok,
            },
        )

        full_name = filter_name(name, self.group_index)
        if full_name in self._groups:
            if not exist_ok:
                raise TensorGroupAlreadyExistsError(name)
            return self[name]

        return self.root._create_group(full_name)

    def rechunk(
        self,
        tensors: Optional[Union[str, List[str]]] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,
    ):
        """Rewrites the underlying chunks to make their sizes optimal.
        This is usually needed in cases where a lot of updates have been made to the data.

        Args:
            tensors (str, List[str], Optional): Name/names of the tensors to rechunk.
                If None, all tensors in the dataset are rechunked.
            num_workers (int): The number of workers to use for rechunking. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used for rechunking. Supported values include: 'serial', 'threaded', and 'processed'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar If ``True`` (default).
        """

        if tensors is None:
            tensors = list(self.tensors)
        elif isinstance(tensors, str):
            tensors = [tensors]

        # identity function that rechunks
        @deeplake.compute
        def rechunking(sample_in, samples_out):
            for tensor in tensors:
                samples_out[tensor].extend(sample_in[tensor])

        rechunking().eval(
            self,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
            skip_ok=True,
            extend_only=True,
            disable_label_sync=True,
            disable_rechunk=True,
        )

    # the below methods are used by cloudpickle dumps
    def __origin__(self):
        return None

    def __values__(self):
        return None

    def __type__(self):
        return None

    def __union_params__(self):
        return None

    def __tuple_params__(self):
        return None

    def __result__(self):
        return None

    def __args__(self):
        return None

    def __bool__(self):
        return True

    def _append_or_extend(
        self,
        sample: Dict[str, Any],
        extend: bool = False,
        skip_ok: bool = False,
        append_empty: bool = False,
    ):
        """Append or extend samples to mutliple tensors at once. This method expects all tensors being updated to be of the same length.

        Args:
            extend (bool): Extends if True. Appends if False.
            sample (dict): Dictionary with tensor names as keys and samples as values.
            skip_ok (bool): Skip tensors not in ``sample`` if set to ``True``.
            append_empty (bool): Append empty samples to tensors not specified in ``sample`` if set to ``True``. If True, ``skip_ok`` is ignored.

        Raises:
            KeyError: If any tensor in the dataset is not a key in ``sample`` and ``skip_ok`` is ``False``.
            TensorDoesNotExistError: If tensor in ``sample`` does not exist.
            ValueError: If all tensors being updated are not of the same length.
            NotImplementedError: If an error occurs while writing tiles.
            Exception: Error while attempting to rollback appends.
            SampleAppendingError: Error that occurs when someone tries to append a tensor value directly to the dataset without specifying tensor name.

        """
        tensors = self.tensors
        if isinstance(sample, Dataset):
            sample = sample.tensors
        if not isinstance(sample, dict):
            raise SampleAppendingError()

        skipped_tensors = [k for k in tensors if k not in sample]
        if skipped_tensors and not skip_ok and not append_empty:
            raise KeyError(
                f"Required tensors not provided: {skipped_tensors}. Pass either `skip_ok=True` to skip tensors or `append_empty=True` to append empty samples to unspecified tensors."
            )
        for k in sample:
            if k not in tensors:
                raise TensorDoesNotExistError(k)
        tensors_to_check_length = tensors if append_empty else sample
        if len(set(map(len, (tensors[k] for k in tensors_to_check_length)))) != 1:
            raise ValueError(
                "When appending using Dataset.append or Dataset.extend, all tensors being updated are expected to have the same length."
            )
        if extend:
            sample_lens = set(map(len, sample.values()))
            if sample_lens == {0}:
                return
            if len(sample_lens) > 1 and not append_empty:
                raise ValueError(
                    "All tensors have to be extended to the same length. Specify `append_empty=True` to pad tensors receiving fewer samples."
                )
            max_len = max(sample_lens)
        [f() for f in list(self._update_hooks.values())]
        tensors_appended = []
        with self:
            for k in tensors:
                extend_extra_nones = 0
                if k in sample:
                    v = sample[k]
                    if extend:
                        extend_extra_nones = max(max_len - len(v), 0)
                else:
                    if skip_ok:
                        continue
                    else:
                        if extend:
                            v = [None] * max_len
                        else:
                            v = None
                try:
                    tensor = tensors[k]
                    enc = tensor.chunk_engine.chunk_id_encoder
                    num_chunks = enc.num_chunks
                    num_samples = tensor.meta.length
                    if extend:
                        tensor._extend(v)
                        if extend_extra_nones:
                            tensor._extend([None] * extend_extra_nones)
                    else:
                        tensor._append(v)
                    tensors_appended.append(k)
                except Exception as e:
                    if extend:
                        raise NotImplementedError(
                            "Unable to recover from error while extending multiple tensors with numpy arrays."
                        )
                    new_num_chunks = enc.num_chunks
                    num_chunks_added = new_num_chunks - num_chunks
                    if num_chunks_added > 1:
                        # This is unlikely to happen, i.e the sample passed the validation
                        # steps and tiling but some error occured while writing tiles to chunks
                        raise NotImplementedError(
                            "Unable to recover from error while writing tiles."
                        ) from e
                    elif num_chunks_added == 1:
                        enc._encoded = enc._encoded[:-1]
                        diff = tensor.meta.length - num_samples
                        tensor.meta.update_length(-diff)
                    for k in tensors_appended:
                        try:
                            self[k].pop()
                        except Exception as e2:
                            raise Exception(
                                "Error while attempting to rollback appends"
                            ) from e2
                    raise e

    def extend(
        self,
        samples: Dict[str, Any],
        skip_ok: bool = False,
        append_empty: bool = False,
        ignore_errors: bool = False,
        progressbar: bool = False,
    ):
        """Appends multiple samples (rows) to multiple tensors at once. This method expects all tensors being updated to be of the same length.

        Args:
            samples (Dict[str, Any]): Dictionary with tensor names as keys and data as values. The values can be a sequence (i.e. a list) or a single numpy array (the first axis in the array is treated as the row axis).
            skip_ok (bool): Skip tensors not in ``samples`` if set to True.
            append_empty (bool): Append empty samples to tensors not specified in ``sample`` if set to ``True``. If True, ``skip_ok`` is ignored.
            ignore_errors (bool): Skip samples that cause errors while extending, if set to ``True``.
            progressbar (bool): Displays a progress bar if set to ``True``.

        Raises:
            KeyError: If any tensor in the dataset is not a key in ``samples`` and ``skip_ok`` is ``False``.
            TensorDoesNotExistError: If tensor in ``samples`` does not exist.
            ValueError: If all tensors being updated are not of the same length.
            NotImplementedError: If an error occurs while writing tiles.
            SampleExtendError: If the extend failed while appending a sample.
            Exception: Error while attempting to rollback appends.

        Examples:

            >>> ds = deeplake.empty("../test/test_ds")

            >>> with ds:
            >>>     ds.create_tensor("data")
            >>>     ds.create_tensor("labels", htype = "class_label")
            >>>     ds.create_tensor("images", htype = "image", sample_compression = "jpeg")

            >>>     # This operation will append 4 samples (rows) to the Deep Lake dataset
            >>>     ds.extend({"data": [1, 2, 3, 4],
                               "labels":["table", "chair", "desk", "table"],
                               "images": [deeplake.read("image1.jpg"), deeplake.read("image2.jpg"), deeplake.read("image3.jpg"), deeplake.read("image4.jpg")]
                               })

        """
        extend = False
        if isinstance(samples, Dataset):
            samples = samples.tensors
            extend = True
        elif set(map(type, samples.values())) == {np.ndarray}:
            extend = True
        if not samples:
            return
        n = len(samples[next(iter(samples.keys()))])
        for v in samples.values():
            if len(v) != n:
                sizes = {k: len(v) for (k, v) in samples.items()}
                raise ValueError(
                    f"Incoming samples are not of equal lengths. Incoming sample sizes: {sizes}"
                )
        len_ds = self.__len__(warn=False)
        new_row_ids = list(range(len_ds, len_ds + n))
        [f() for f in list(self._update_hooks.values())]
        if extend:
            if ignore_errors:
                warnings.warn(
                    "`ignore_errors` argument will be ignored while extending with numpy arrays or tensors."
                )
            self._append_or_extend(
                samples,
                extend=True,
                skip_ok=skip_ok,
                append_empty=append_empty,
            )
        else:
            with self:
                if progressbar:
                    indices = tqdm(range(n))
                else:
                    indices = range(n)
                for i in indices:
                    try:
                        self._append_or_extend(
                            {k: v[i] for k, v in samples.items()},
                            extend=False,
                            skip_ok=skip_ok,
                            append_empty=append_empty,
                        )
                    except Exception as e:
                        if ignore_errors:
                            continue
                        else:
                            if isinstance(e, SampleAppendError):
                                raise SampleExtendError(str(e)) from e.__cause__
                            raise e
        index_maintenance.index_operation_dataset(
            self, dml_type=_INDEX_OPERATION_MAPPING["ADD"], rowids=new_row_ids
        )

    @invalid_view_op
    def append(
        self,
        sample: Dict[str, Any],
        skip_ok: bool = False,
        append_empty: bool = False,
    ):
        """Append a single sample (row) to multiple tensors at once.

        Args:
            sample (dict): Dictionary with tensor names as keys and samples as values.
            skip_ok (bool): Skip tensors not in ``sample`` if set to ``True``.
            append_empty (bool): Append empty samples to tensors not specified in ``sample`` if set to ``True``. If True, ``skip_ok`` is ignored.

        Raises:
            KeyError: If any tensor in the dataset is not a key in ``sample`` and ``skip_ok`` is ``False``.
            TensorDoesNotExistError: If tensor in ``sample`` does not exist.
            ValueError: If all tensors being updated are not of the same length.
            NotImplementedError: If an error occurs while writing tiles.
            Exception: Error while attempting to rollback appends.
            SampleAppendingError: Error that occurs when someone tries to append a tensor value directly to the dataset without specifying tensor name.

        Examples:

            >>> ds = deeplake.empty("../test/test_ds")

            >>> with ds:
            >>>     ds.create_tensor('data')
            >>>     ds.create_tensor('labels')

            >>>     # This operation will append 1 sample (row) to the Deep Lake dataset
            >>>     ds.append({"data": 1, "labels": "table"})

        """
        new_row_ids = [self.__len__(warn=False)]
        self._append_or_extend(
            sample,
            extend=False,
            skip_ok=skip_ok,
            append_empty=append_empty,
        )
        index_maintenance.index_operation_dataset(
            self, dml_type=_INDEX_OPERATION_MAPPING["ADD"], rowids=new_row_ids
        )

    def update(self, sample: Dict[str, Any]):
        """Update existing samples in the dataset with new values.

        Examples:

            >>> ds[0].update({"images": deeplake.read("new_image.png"), "labels": 1})

            >>> new_images = [deeplake.read(f"new_image_{i}.png") for i in range(3)]
            >>> ds[:3].update({"images": new_images, "labels": [1, 2, 3]})

        Args:
            sample (dict): Dictionary with tensor names as keys and samples as values.

        Raises:
            ValueError: If partial update of a sample is attempted.
            Exception: Error while attempting to rollback updates.
        """
        if len(self.index) > 1:
            raise ValueError(
                "Cannot make partial updates to samples using `ds.update`. Use `ds.tensor[index] = value` instead."
            )

        def get_sample_from_engine(
            engine, idx, is_link, compression, dtype, decompress
        ):
            # tiled data will always be decompressed
            decompress = decompress or engine._is_tiled_sample(idx)
            if is_link:
                creds_key = engine.creds_key(idx)
                item = engine.get_path(idx)
                return LinkedSample(item, creds_key)
            item = engine.get_single_sample(idx, self.index, decompress=decompress)
            shape = engine.read_shape_for_sample(idx)
            return engine._get_sample_object(
                item, shape, compression, dtype, decompress
            )

        # remove update hooks from view base so that the view is not invalidated
        if self._view_base:
            saved_update_hooks = self._view_base._update_hooks
            self._view_base._update_hooks = {}
        idx = self.index.values[0].value
        with self:
            saved = defaultdict(list)
            try:
                for k, v in sample.items():
                    tensor_meta = self[k].meta
                    dtype = tensor_meta.dtype
                    sample_compression = tensor_meta.sample_compression
                    chunk_compression = tensor_meta.chunk_compression

                    compression = sample_compression or chunk_compression

                    engine = self[k].chunk_engine

                    decompress = chunk_compression is not None or engine.is_text_like

                    for idx in self.index.values[0].indices(self[k].num_samples):
                        if tensor_meta.is_sequence:
                            old_sample = []
                            for i in range(*engine.sequence_encoder[idx]):
                                item = get_sample_from_engine(
                                    engine,
                                    i,
                                    tensor_meta.is_link,
                                    compression,
                                    dtype,
                                    decompress,
                                )
                                old_sample.append(item)
                        else:
                            old_sample = get_sample_from_engine(
                                engine,
                                idx,
                                tensor_meta.is_link,
                                compression,
                                dtype,
                                decompress,
                            )

                        saved[k].append(old_sample)
                    self[k] = v
                # Regenerate Index
                index_maintenance.index_operation_dataset(
                    self,
                    dml_type=_INDEX_OPERATION_MAPPING["UPDATE"],
                    rowids=list(self.index.values[0].indices(self.__len__(warn=False))),
                )
            except Exception as e:
                for k, v in saved.items():
                    # squeeze
                    if len(v) == 1:
                        v = v[0]
                    try:
                        self[k] = v
                    except Exception as e2:
                        raise Exception(
                            "Error while attempting to rollback updates"
                        ) from e2
                # in case of error, regenerate index again to avoid index corruption
                index_maintenance.index_operation_dataset(
                    self,
                    dml_type=_INDEX_OPERATION_MAPPING["UPDATE"],
                    rowids=list(self.index.values[0].indices(self.__len__(warn=False))),
                )
                raise e
            finally:
                # restore update hooks
                if self._view_base:
                    self._view_base._update_hooks = saved_update_hooks

    def _view_hash(self) -> str:
        """Generates a unique hash for a filtered dataset view."""
        return hash_inputs(
            self.path,
            *[e.value for e in self.index.values],
            self.pending_commit_id,
            getattr(self, "_query", None),
            getattr(self, "_tql_query", None),
        )

    def _get_view_info(
        self,
        id: Optional[str] = None,
        message: Optional[str] = None,
        copy: bool = False,
    ):
        if self.has_head_changes and not self.is_optimized:
            raise DatasetViewSavingError(
                "The dataset's HEAD node has uncommitted changes. Please create a commit on"
                " the dataset object [ds.commit(<insert optional message>)] prior to saving the view."
            )
        commit_id = self.commit_id
        tm = getattr(self, "_created_at", time())
        id = self._view_hash() if id is None else id
        info = {
            "id": id,
            "virtual-datasource": not copy,
            "source-dataset": self.path,
            "source-dataset-version": commit_id,
            "created_at": tm,
        }
        if message is not None:
            info["message"] = message
        query = getattr(self, "_query", None)
        if query:
            info["query"] = query
            info["source-dataset-index"] = getattr(self, "_source_ds_idx", None)
        tql_query = getattr(self, "_tql_query", None)
        if tql_query:
            info["tql_query"] = tql_query
            info["source-dataset-index"] = getattr(self, "_source_ds_idx", None)
        if not (query or tql_query):
            info["source-dataset-index"] = self.index.to_json()
        return info

    def _lock_queries_json(self):
        class _LockQueriesJson:
            def __enter__(self2):
                storage = self.base_storage
                self2.storage_read_only = storage.read_only
                if self._locked_out:
                    # Ignore storage level lock since we have file level lock
                    storage.read_only = False
                lock = Lock(storage, get_queries_lock_key())
                lock.acquire(timeout=10)
                self2.lock = lock

            def __exit__(self2, *_, **__):
                self2.lock.release()
                self.base_storage.read_only = self2.storage_read_only

        return _LockQueriesJson()

    def _write_queries_json(self, data: dict):
        read_only = self.base_storage.read_only
        self.base_storage.disable_readonly()
        try:
            self.base_storage[get_queries_key()] = json.dumps(data).encode("utf-8")
        finally:
            if read_only:
                self.base_storage.enable_readonly()

    def _append_to_queries_json(self, info: dict):
        with self._lock_queries_json():
            qjson = self._read_queries_json()
            idx = None
            for i in range(len(qjson)):
                if qjson[i]["id"] == info["id"]:
                    idx = i
                    break
            if idx is None:
                qjson.append(info)
            else:
                qjson[idx] = info
            self._write_queries_json(qjson)

    def _read_queries_json(self) -> list:
        try:
            return json.loads(self.base_storage[get_queries_key()].decode("utf-8"))
        except KeyError:
            return []

    def _read_view_info(self, id: str):
        for info in self._read_queries_json():
            if info["id"] == id:
                return info
        raise KeyError(f"View with id {id} not found.")

    def _write_vds(
        self,
        vds,
        info: dict,
        copy: Optional[bool] = False,
        tensors: Optional[List[str]] = None,
        num_workers: Optional[int] = 0,
        scheduler: str = "threaded",
        ignore_errors: bool = False,
        unlink=True,
    ):
        """Writes the indices of this view to a vds."""
        vds._allow_view_updates = True
        try:
            with vds:
                if copy:
                    self._copy(
                        vds,
                        tensors=tensors,
                        num_workers=num_workers,
                        scheduler=scheduler,
                        unlink=unlink,
                        create_vds_index_tensor=True,
                        ignore_errors=ignore_errors,
                    )
                else:
                    vds.create_tensor(
                        "VDS_INDEX",
                        dtype="uint64",
                        create_shape_tensor=False,
                        create_id_tensor=False,
                        create_sample_info_tensor=False,
                    ).extend(
                        np.array(
                            tuple(self.index.values[0].indices(self.num_samples)),
                            dtype="uint64",
                        ),
                        progressbar=True,
                    )
                    info["first-index-subscriptable"] = self.index.subscriptable_at(0)
                    if len(self.index) > 1:
                        info["sub-sample-index"] = Index(
                            self.index.values[1:]
                        ).to_json()
                vds.info.update(info)
        finally:
            try:
                delattr(vds, "_allow_view_updates")
            except AttributeError:  # Attribute already deleted by _copy()
                pass

    def _save_view_in_subdir(
        self,
        id: Optional[str],
        message: Optional[str],
        copy: bool,
        tensors: Optional[List[str]],
        num_workers: int,
        scheduler: str,
        ignore_errors: bool,
        overwrite: bool = False,
    ):
        """Saves this view under ".queries" subdirectory of same storage."""
        if not overwrite:
            existing_views = self.get_views()
            for v in existing_views:
                if v.id == id:
                    raise DatasetViewSavingError(
                        f"View with id {id} already exists. Use a different id or delete the existing view."
                    )

        info = self._get_view_info(id, message, copy)
        hash = info["id"]
        # creating sub-view of optimized view
        if self.is_optimized:
            view_entry = self._view_entry
            ds = view_entry._src_ds.no_view_dataset

            if copy:
                view_info = view_entry.info
                info["source-dataset"] = ds.path
                info["source-dataset-version"] = view_info["source-dataset-version"]
                if "source-dataset-index" in view_info:
                    original_idx = Index.from_json(view_info["source-dataset-index"])
                    combined_idx = original_idx[self.index]
                    info["source-dataset-index"] = combined_idx.to_json()
            else:
                info["source-dataset-version"] = (
                    info["source-dataset-version"] or FIRST_COMMIT_ID
                )
        else:
            ds = self
        path = f".queries/{hash}"
        vds = ds._sub_ds(path, empty=True, verbose=False)
        self._write_vds(vds, info, copy, tensors, num_workers, scheduler, ignore_errors)
        ds._append_to_queries_json(info)
        return vds

    def _save_view_in_path(
        self,
        path: str,
        id: Optional[str],
        message: Optional[str],
        copy: bool,
        tensors: Optional[List[str]],
        num_workers: int,
        scheduler: str,
        ignore_errors: bool,
        overwrite: bool = False,
        **ds_args,
    ):
        """Saves this view at a given dataset path"""
        if os.path.abspath(path) == os.path.abspath(self.path):
            raise DatasetViewSavingError("Rewriting parent dataset is not allowed.")
        try:
            vds = deeplake.empty(path, overwrite=overwrite, **ds_args)
        except Exception as e:
            raise DatasetViewSavingError from e
        info = self._get_view_info(id, message, copy)
        self._write_vds(vds, info, copy, tensors, num_workers, scheduler, ignore_errors)
        return vds

    def save_view(
        self,
        message: Optional[str] = None,
        path: Optional[Union[str, pathlib.Path]] = None,
        id: Optional[str] = None,
        optimize: bool = False,
        tensors: Optional[List[str]] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        verbose: bool = True,
        ignore_errors: bool = False,
        overwrite: bool = False,
        **ds_args,
    ) -> str:
        """Saves a dataset view as a virtual dataset (VDS)

        Examples:

            >>> # Save to specified path
            >>> vds_path = ds[:10].save_view(path="views/first_10", id="first_10")
            >>> vds_path
            views/first_10

            >>> # Path unspecified
            >>> vds_path = ds[:100].save_view(id="first_100", message="first 100 samples")
            >>> # vds_path = path/to/dataset

            >>> # Random id
            >>> vds_path = ds[:100].save_view()
            >>> # vds_path = path/to/dataset/.queries/92f41922ed0471ec2d27690b7351fc96bea060e6c5ee22b14f7ffa5f291aa068

            See :func:`Dataset.get_view` to learn how to load views by id.
            These virtual datasets can also be loaded from their path like normal datasets.

        Args:
            message (Optional, str): Custom user message.
            path (Optional, str, pathlib.Path): - The VDS will be saved as a standalone dataset at the specified path.
                - If not specified, the VDS is saved under ``.queries`` subdirectory of the source dataset's storage.
            id (Optional, str): Unique id for this view. Random id will be generated if not specified.
            optimize (bool):
                - If ``True``, the dataset view will be optimized by copying and rechunking the required data. This is necessary to achieve fast streaming speeds when training models using the dataset view. The optimization process will take some time, depending on the size of the data.
                - You can also choose to optimize the saved view later by calling its :meth:`ViewEntry.optimize` method.
            tensors (List, optional): Names of tensors (and groups) to be copied. If not specified all tensors are copied.
            num_workers (int): Number of workers to be used for optimization process. Applicable only if ``optimize=True``. Defaults to 0.
            scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded', and 'processed'. Only applicable if ``optimize=True``. Defaults to 'threaded'.
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.
            ignore_errors (bool): Skip samples that cause errors while saving views. Only applicable if ``optimize=True``. Defaults to ``False``.
            overwrite (bool): If true, any existing view with the same id is silently overwritten. If false, an exception is thrown if a view with the same the id exists. Defaults to ``False``.
            ds_args (dict): Additional args for creating VDS when path is specified. (See documentation for :func:`deeplake.dataset()`)

        Returns:
            str: Path to the saved VDS.

        Raises:
            ReadOnlyModeError: When attempting to save a view inplace and the user doesn't have write access.
            DatasetViewSavingError: If HEAD node has uncommitted changes.
            TypeError: If ``id`` is not of type ``str``.

        Note:
            Specifying ``path`` makes the view external. External views cannot be accessed using the parent dataset's :func:`Dataset.get_view`,
            :func:`Dataset.load_view`, :func:`Dataset.delete_view` methods. They have to be loaded using :func:`deeplake.load`.
        """

        deeplake_reporter.feature_report(
            feature_name="save_view",
            parameters={
                "id": id,
                "optimize": optimize,
                "tensors": tensors,
                "num_workers": num_workers,
                "scheduler": scheduler,
                "verbose": verbose,
            },
        )

        if id is not None and not isinstance(id, str):
            raise TypeError(f"id {id} is of type {type(id)}, expected `str`.")
        return self._save_view(
            path,
            id,
            message or self._query_string,
            optimize,
            tensors,
            num_workers,
            scheduler,
            verbose,
            False,
            ignore_errors,
            overwrite,
            **ds_args,
        )

    def _save_view(
        self,
        path: Optional[Union[str, pathlib.Path]] = None,
        id: Optional[str] = None,
        message: Optional[str] = None,
        optimize: bool = False,
        tensors: Optional[List[str]] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        verbose: bool = True,
        _ret_ds: bool = False,
        ignore_errors: bool = False,
        overwrite: bool = False,
        **ds_args,
    ) -> Union[str, Any]:
        """Saves a dataset view as a virtual dataset (VDS)

        Args:
            path (Optional, str, pathlib.Path): If specified, the VDS will saved as a standalone dataset at the specified path. If not,
                the VDS is saved under `.queries` subdirectory of the source dataset's storage.
            id (Optional, str): Unique id for this view.
            message (Optional, message): Custom user message.
            optimize (bool): Whether the view should be optimized by copying the required data. Default False.
            tensors (Optional, List[str]): Tensors to be copied if `optimize` is True. By default all tensors are copied.
            num_workers (int): Number of workers to be used if `optimize` is True.
            scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded', and 'processed'.
                Only applicable if ``optimize=True``. Defaults to 'threaded'.
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.
            _ret_ds (bool): If ``True``, the VDS is retured as such without converting it to a view. If ``False``, the VDS path is returned.
                Default False.
            ignore_errors (bool): Skip samples that cause errors while saving views. Only applicable if ``optimize=True``. Defaults to ``False``.
            overwrite (bool): If true, any existing view with the same id is silently overwritten. If false, an exception is thrown if a view with the same the id exists. Defaults to ``False``.
            ds_args (dict): Additional args for creating VDS when path is specified. (See documentation for `deeplake.dataset()`)

        Returns:
            If ``_ret_ds`` is ``True``, the VDS is returned, else path to the VDS is returned.

        Raises:
            ReadOnlyModeError: When attempting to save a view inplace and the user doesn't have write access.
            NotImplementedError: When attempting to save in-memory datasets.
        """

        path = convert_pathlib_to_string_if_needed(path)
        ds_args["verbose"] = False
        vds = None
        if path is None and hasattr(self, "_vds"):
            vds = self._vds
            vds_id = vds.info["id"]
            if id is not None and vds_id != id:
                vds = None
                warnings.warn(
                    f"This view is already saved with id '{vds_id}'. A copy of this view will be created with the provided id '{id}'"
                )
        base = self._view_base or self
        if not base._read_only:
            base.flush()
        if vds is None:
            if path is None:
                if isinstance(self, MemoryProvider):
                    raise NotImplementedError(
                        "Saving views inplace is not supported for in-memory datasets."
                    )

                if self.read_only and not base._locked_out and not self.is_optimized:
                    if isinstance(self, deeplake.core.dataset.DeepLakeCloudDataset):
                        try:
                            with self._temp_write_access():
                                vds = self._save_view_in_subdir(
                                    id,
                                    message,
                                    optimize,
                                    tensors,
                                    num_workers,
                                    scheduler,
                                    ignore_errors,
                                    overwrite,
                                )
                        except ReadOnlyModeError as e:
                            raise ReadOnlyModeError(
                                "Cannot save a view in this dataset because you are not a member of its organization."
                                "Please specify a `path` in order to save the view at a custom location."
                            ) from e
                    else:
                        raise ReadOnlyModeError(
                            "Cannot save view in read only dataset. Specify a path to save the view in a different location."
                        )
                else:
                    vds = self._save_view_in_subdir(
                        id,
                        message,
                        optimize,
                        tensors,
                        num_workers,
                        scheduler,
                        ignore_errors,
                        overwrite,
                    )
            else:
                vds = self._save_view_in_path(
                    path,
                    id,
                    message,
                    optimize,
                    tensors,
                    num_workers,
                    scheduler,
                    ignore_errors,
                    overwrite,
                    **ds_args,
                )
        if verbose and self.verbose:
            log_visualizer_link(vds.path, self.path)
        if _ret_ds:
            return vds
        return vds.path

    def _get_view(self, inherit_creds=True, creds: Optional[Dict] = None):
        """Returns a view for this VDS. Only works if this Dataset is a virtual dataset.

        Returns:
            A view of the source dataset based on the indices from VDS.

        Args:
            inherit_creds (bool): Whether to inherit creds from the parent dataset in which this vds is stored. Default True.
            creds (optional, Dict): Creds for the source dataset. Used only if inherit_creds is False.

        Raises:
            Exception: If this is not a VDS.
        """

        try:
            commit_id = self.info["source-dataset-version"]
        except KeyError:
            raise Exception("Dataset._get_view() works only for virtual datasets.")
        ds = (
            self._parent_dataset[Index()]
            if (inherit_creds and self._parent_dataset)
            else deeplake.load(
                self.info["source-dataset"],
                verbose=False,
                creds=creds,
                read_only=True,
                token=self._token,
            )
        )

        ds.index = Index()
        ds.version_state = ds.version_state.copy()
        ds._checkout(commit_id, verbose=False)
        first_index_subscriptable = self.info.get("first-index-subscriptable", True)
        if first_index_subscriptable:
            index_entries = [IndexEntry(self.VDS_INDEX.numpy().reshape(-1).tolist())]
        else:
            index_entries = [IndexEntry(int(self.VDS_INDEX.numpy()))]
        sub_sample_index = self.info.get("sub-sample-index")
        if sub_sample_index:
            index_entries += Index.from_json(sub_sample_index).values
        ret = ds[Index(index_entries)]
        ret._vds = self
        return ret

    def _get_empty_vds(
        self,
        vds_path: Optional[Union[str, pathlib.Path]] = None,
        query: Optional[str] = None,
        **vds_args,
    ):
        """Returns an empty VDS with this dataset as the source dataset. Internal.

        Args:
            vds_path (Optional, str, pathlib.Path): If specified, the vds will be sved at this path. Else the vds will be saved under `.queries` subdirectory.
            query (Optional, str): Query string associated with this view.
            vds_args (dict): Additional args for creating vds when path is specified.

        Returns:
            Empty VDS with this dataset as the source dataset.
        """
        view = self[:0]
        vds_path = convert_pathlib_to_string_if_needed(vds_path)
        if query:
            view._query = query
        return view._save_view(vds_path, _ret_ds=True, **vds_args)

    def get_views(self, commit_id: Optional[str] = None) -> List[ViewEntry]:
        """Returns list of views stored in this Dataset.

        Args:
            commit_id (str, optional): - Commit from which views should be returned.
                - If not specified, views from all commits are returned.

        Returns:
            List[ViewEntry]: List of :class:`ViewEntry` instances.
        """
        queries = self._read_queries_json()
        if commit_id is not None:
            queries = filter(
                lambda x: x["source-dataset-version"] == commit_id, queries
            )
        return list(map(partial(ViewEntry, dataset=self), queries))

    def get_view(self, id: str) -> ViewEntry:
        """Returns the dataset view corresponding to ``id``.

        Examples:
            >>> # save view
            >>> ds[:100].save_view(id="first_100")
            >>> # load view
            >>> first_100 = ds.get_view("first_100").load()
            >>> # 100
            >>> print(len(first_100))

            See :func:`Dataset.save_view` to learn more about saving views.

        Args:
            id (str): id of required view.

        Returns:
            ViewEntry

        Raises:
            KeyError: If no such view exists.
        """
        queries = self._read_queries_json()
        for q in queries:
            if q["id"] == id:
                return ViewEntry(q, self)
        raise KeyError(f"No view with id {id} found in the dataset.")

    def load_view(
        self,
        id: str,
        optimize: Optional[bool] = False,
        tensors: Optional[List[str]] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: Optional[bool] = True,
    ):
        """Loads the view and returns the :class:`~deeplake.core.dataset.dataset.Dataset` by id. Equivalent to ds.get_view(id).load().

        Args:
            id (str): id of the view to be loaded.
            optimize (bool): If ``True``, the dataset view is optimized by copying and rechunking the required data before loading. This is
                necessary to achieve fast streaming speeds when training models using the dataset view. The optimization process will
                take some time, depending on the size of the data.
            tensors (Optional, List[str]): Tensors to be copied if `optimize` is True. By default all tensors are copied.
            num_workers (int): Number of workers to be used for the optimization process. Only applicable if `optimize=True`. Defaults to 0.
            scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded', and 'processed'.
                Only applicable if `optimize=True`. Defaults to 'threaded'.
            progressbar (bool): Whether to use progressbar for optimization. Only applicable if `optimize=True`. Defaults to True.

        Returns:
            Dataset: The loaded view.

        Raises:
            KeyError: if view with given id does not exist.
        """
        deeplake_reporter.feature_report(
            feature_name="load_view",
            parameters={
                "id": id,
                "optimize": optimize,
                "tensors": tensors,
                "num_workers": num_workers,
                "scheduler": scheduler,
            },
        )

        view = self.get_view(id)
        if optimize:
            return view.optimize(
                tensors=tensors,
                num_workers=num_workers,
                scheduler=scheduler,
                progressbar=progressbar,
            ).load()
        return view.load()

    def delete_view(self, id: str):
        """Deletes the view with given view id.

        Args:
            id (str): Id of the view to delete.

        Raises:
            KeyError: if view with given id does not exist.
        """

        deeplake_reporter.feature_report(
            feature_name="delete_view",
            parameters={"id": id},
        )

        try:
            with self._lock_queries_json():
                qjson = self._read_queries_json()
                for i, q in enumerate(qjson):
                    if q["id"] == id:
                        qjson.pop(i)
                        self.base_storage.subdir(
                            ".queries/" + (q.get("path") or q["id"])
                        ).clear()
                        self._write_queries_json(qjson)
                        return
        # not enough permissions to acquire lock
        except TokenPermissionError:
            pass

        raise KeyError(f"No view with id {id} found in the dataset.")

    def _sub_ds(
        self,
        path,
        empty=False,
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        read_only=None,
        lock=True,
        verbose=True,
        token=None,
    ):
        """Loads a nested dataset. Internal.

        Args:
            path (str): Path to sub directory
            empty (bool): If ``True``, all contents of the sub directory is cleared before initializing the sub dataset.
            memory_cache_size (int): Memory cache size for the sub dataset.
            local_cache_size (int): Local storage cache size for the sub dataset.
            read_only (bool): Loads the sub dataset in read only mode if ``True``. Default ``False``.
            lock (bool): Whether the dataset should be locked for writing. Only applicable for S3, Deep Lake and GCS datasets. No effect if ``read_only=True``.
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.
            token (Optional[str]): Token of source dataset.

        Returns:
            Sub dataset

        Note:
            Virtual datasets are returned as such, they are not converted to views.
        """
        sub_storage = self.base_storage.subdir(path, read_only=read_only)

        if empty:
            sub_storage.clear()

        if self.path.startswith("hub://"):
            path = posixpath.join(self.path, path)
            cls = deeplake.core.dataset.DeepLakeCloudDataset
        else:
            path = sub_storage.root
            cls = deeplake.core.dataset.Dataset

        ret = cls(
            generate_chain(
                sub_storage,
                memory_cache_size * MB,
                local_cache_size * MB,
            ),
            path=path,
            token=token,
            read_only=read_only,
            lock=lock,
            verbose=verbose,
        )
        ret._parent_dataset = self
        return ret

    def _link_tensors(
        self,
        src: str,
        dest: str,
        extend_f: str,
        update_f: Optional[str] = None,
        flatten_sequence: Optional[bool] = None,
    ):
        """Internal. Links a source tensor to a destination tensor. Appends / updates made to the source tensor will be reflected in the destination tensor.

        Args:
            src (str): Name of the source tensor.
            dest (str): Name of the destination tensor.
            extend_f (str): Name of the linked tensor transform to be used for extending the destination tensor. This transform should be defined in `deeplake.core.tensor_link` module.
            update_f (str): Name of the linked tensor transform to be used for updating items in the destination tensor. This transform should be defined in `deeplake.core.tensor_link` module.
            flatten_sequence (bool, Optional): Whether appends and updates should be done per item or per sequence if the source tensor is a sequence tensor.

        Raises:
            TensorDoesNotExistError: If source or destination tensors do not exist in this dataset.
            ValueError: If source tensor is a sequence tensor and `flatten_sequence` argument is not specified.
        """
        assert self._is_root()
        tensors = self._tensors()
        if src not in tensors:
            raise TensorDoesNotExistError(src)
        if dest not in tensors:
            raise TensorDoesNotExistError(dest)
        src_tensor = self[src]
        dest_key = self.version_state["tensor_names"][dest]
        if flatten_sequence is None:
            if src_tensor.is_sequence:
                raise ValueError(
                    "`flatten_sequence` arg must be specified when linking a sequence tensor."
                )
            flatten_sequence = False
        src_tensor.meta.add_link(dest_key, extend_f, update_f, flatten_sequence)
        self.storage.maybe_flush()

    def _resolve_tensor_list(self, keys: List[str], root: bool = False) -> List[str]:
        ret = []
        for k in keys:
            fullpath = k if root else posixpath.join(self.group_index, k)
            if (
                self.version_state["tensor_names"].get(fullpath)
                in self.version_state["full_tensors"]
            ):
                ret.append(k)
            else:
                enabled_tensors = self.enabled_tensors
                if fullpath[-1] != "/":
                    fullpath = fullpath + "/"
                hidden = self.meta.hidden_tensors
                ret += filter(
                    lambda t: t.startswith(fullpath)
                    and t not in hidden
                    and (enabled_tensors is None or t in enabled_tensors),
                    self.version_state["tensor_names"],
                )
        return ret

    def _copy(
        self,
        dest: Union[str, pathlib.Path],
        runtime: Optional[Dict] = None,
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        creds=None,
        token=None,
        num_workers: int = 0,
        scheduler="threaded",
        progressbar=True,
        public: bool = False,
        unlink: bool = False,
        create_vds_index_tensor: bool = False,
        ignore_errors: bool = False,
        verbose: bool = True,
    ):
        if isinstance(dest, str):
            path = dest
        else:
            path = dest.path

        dest_ds = deeplake.api.dataset.dataset._like(
            dest,
            self,
            runtime=runtime,
            tensors=tensors,
            creds=creds,
            token=token,
            overwrite=overwrite,
            public=public,
            unlink=(
                [
                    t
                    for t in self.tensors
                    if (
                        self.tensors[t].base_htype != "video"
                        or deeplake.constants._UNLINK_VIDEOS
                    )
                ]
                if unlink
                else False
            ),
            verbose=verbose,
        )

        # dest_ds needs link creds of source dataset for copying linked tensors
        dest_ds.link_creds = LinkCreds()
        dest_ds.link_creds.__setstate__(self.link_creds.__getstate__())
        save_link_creds(dest_ds.link_creds, dest_ds.storage)

        def _is_unlink_tensor(tensor):
            if (
                unlink
                and tensor.is_link
                and (tensor.base_htype != "video" or deeplake.constants._UNLINK_VIDEOS)
            ):
                return True

        # If we have to unlink any tensor, we will use sample-by-sample append implementation (_copy_tensor_append)
        # Otherwise, we will use extend-by-whole-slice implementation (_copy_tensor_extend)
        extend_only = not any(
            _is_unlink_tensor(self[tensor_name]) for tensor_name in dest_ds.tensors
        )

        def _copy_tensor_extend(sample_in, sample_out):
            for tensor_name in dest_ds.tensors:
                sample_out[tensor_name].extend(sample_in[tensor_name])

        def _copy_tensor_append(sample_in, sample_out):
            for tensor_name in dest_ds.tensors:
                src = sample_in[tensor_name]
                if _is_unlink_tensor(src):
                    if len(sample_in.index) > 1:
                        sample_out[tensor_name].append(src)
                    else:
                        idx = sample_in.index.values[0].value
                        sample_out[tensor_name].append(
                            src.chunk_engine.get_deeplake_read_sample(idx)
                        )
                else:
                    sample_out[tensor_name].append(src)

        if not self.index.subscriptable_at(0):
            old_first_index = self.index.values[0]
            new_first_index = IndexEntry(
                slice(old_first_index.value, old_first_index.value + 1)
            )
            self.index.values[0] = new_first_index
            reset_index = True
        else:
            reset_index = False
        try:
            deeplake.compute(
                _copy_tensor_extend if extend_only else _copy_tensor_append,
                name="copy transform",
            )().eval(
                self,
                dest_ds,
                num_workers=num_workers,
                scheduler=scheduler,
                progressbar=progressbar,
                skip_ok=True,
                check_lengths=False,
                ignore_errors=ignore_errors,
                disable_label_sync=True,
                extend_only=extend_only,
            )

            dest_ds.flush()
            if create_vds_index_tensor:
                with dest_ds:
                    try:
                        dest_ds._allow_view_updates = True
                        dest_ds.create_tensor(
                            "VDS_INDEX",
                            dtype=np.uint64,
                            hidden=True,
                            create_shape_tensor=False,
                            create_id_tensor=False,
                            create_sample_info_tensor=False,
                        )
                        dest_ds.VDS_INDEX.extend(list(self.sample_indices))
                    finally:
                        delattr(dest_ds, "_allow_view_updates")
        finally:
            if reset_index:
                dest_ds.meta.default_index = Index([IndexEntry(0)]).to_json()
                dest_ds.meta.is_dirty = True
                dest_ds.flush()
                dest_ds = dest_ds[0]
                self.index.values[0] = old_first_index

            # Actual credentials should be removed from dest_ds after copy
            dest_ds.link_creds = None
            dest_ds._load_link_creds()
        return dest_ds

    def copy(
        self,
        dest: Union[str, pathlib.Path],
        runtime: Optional[dict] = None,
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        creds=None,
        token=None,
        num_workers: int = 0,
        scheduler="threaded",
        progressbar=True,
        public: bool = False,
    ):
        """Copies this dataset or dataset view to ``dest``. Version control history is not included, and this operation copies data from the latest commit on the main branch.

        Args:
            dest (str, pathlib.Path): Destination dataset or path to copy to. If a Dataset instance is provided, it is expected to be empty.
            tensors (List[str], optional): Names of tensors (and groups) to be copied. If not specified all tensors are copied.
            runtime (dict): Parameters for Activeloop DB Engine. Only applicable for hub://... paths.
            overwrite (bool): If ``True`` and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            creds (dict, Optional): creds required to create / overwrite datasets at `dest`.
            token (str, Optional): token used to for fetching credentials to `dest`.
            num_workers (int): The number of workers to use for copying. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used for copying. Supported values include: 'serial', 'threaded', and 'processed'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar If ``True`` (default).
            public (bool): Defines if the dataset will have public access. Applicable only if Deep Lake cloud storage is used and a new Dataset is being created. Defaults to False.

        Returns:
            Dataset: New dataset object.

        Raises:
            DatasetHandlerError: If a dataset already exists at destination path and overwrite is False.
        """

        deeplake_reporter.feature_report(
            feature_name="copy",
            parameters={
                "tensors": tensors,
                "overwrite": overwrite,
                "num_workers": num_workers,
                "scheduler": scheduler,
                "progressbar": progressbar,
                "public": public,
            },
        )

        return self._copy(
            dest,
            runtime,
            tensors,
            overwrite,
            creds,
            token,
            num_workers,
            scheduler,
            progressbar,
            public,
        )

    @invalid_view_op
    @spinner
    def reset(self, force: bool = False):
        """Resets the uncommitted changes present in the branch.

        Note:
            The uncommitted data is deleted from underlying storage, this is not a reversible operation.
        """

        deeplake_reporter.feature_report(
            feature_name="reset",
            parameters={"force": force},
        )

        storage, version_state = self.storage, self.version_state
        if version_state["commit_node"].children:
            print("You are not at the head node of the branch, cannot reset.")
            return
        if not self.has_head_changes and not force:
            print("There are no uncommitted changes on this branch.")
            return

        if self.commit_id is None:
            storage.clear()
            self._populate_meta()
            load_meta(self)
        else:
            parent_commit_id = self.commit_id
            reset_commit_id = self.pending_commit_id

            # checkout to get list of tensors in previous commit, needed for copying metas and create_commit_chunk_set
            self.checkout(parent_commit_id)

            new_commit_id = replace_head(storage, version_state, reset_commit_id)

            self.checkout(new_commit_id)

    def fix_vc(self):
        """Rebuilds version control info. To be used when the version control info is corrupted."""
        version_info = rebuild_version_info(self.storage)
        self.version_state["commit_node_map"] = version_info["commit_node_map"]
        self.version_state["branch_commit_map"] = version_info["branch_commit_map"]

    def connect(
        self,
        creds_key: Optional[str] = None,
        dest_path: Optional[str] = None,
        org_id: Optional[str] = None,
        ds_name: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """Connect a Deep Lake dataset stored in your cloud to the Deep Lake App. This enabables you to visualize the dataset in the Deep Lake App and have access to the full Deep Lake feature set.

        Examples:
            >>> # Load (or create) a could dataset using Deep Lake Managed Credentials from your org
            >>> ds = deeplake.load("s3://bucket/dataset", creds={"creds_key": "managed_creds_key"}, org_id="my_org_id")

            >>> # Connect the dataset to the Deep Lake App using the same managed credentials and organization as above
            >>> ds.connect()

            >>> # Or specify an alternative path and/or managed credentials to connect the dataset
            >>> ds.connect(dest_path="hub://my_org/dataset", creds_key = "different_creds_key")

            >>> # You can opt not to use Deep Lake Managed Credentials to load/create the dataset, but you must use Managed Credentials to connect the dataset to Deep Lake
            >>> ds = deeplake.load("s3://bucket/dataset", creds = {"aws_access_key_id": ..., ...})
            >>> ds.connect(org_id="my_org_id", creds_key = "managed_creds_key")


        Args:
            creds_key (str): The managed credentials to be used for accessing the source path. Optional if the dataset was orginally loaded with a creds_key
            dest_path (str, optional): The full path to which the connected Deep Lake dataset will reside. Can be:
                a Deep Lake path like ``hub://<org_id>/<dataset_name>``
            org_id (str, optional): The organization to which the connected Deep Lake dataset will be added. The dataset name will be the same as parent folder for the dataset in the cloud storage location.
            ds_name (str, optional): The name of the connected Deep Lake dataset. Will be infered from ``dest_path`` or storage location path if not provided.
            token (str, optional): Activeloop token used to fetch the managed credentials.

        Raises:
            InvalidSourcePathError: If the dataset's path is not a valid s3, gcs or azure path.
            InvalidDestinationPathError: If ``dest_path``, or ``org_id`` and ``ds_name`` do not form a valid Deep Lake path.
            TokenPermissionError: If the user does not have permission to create a dataset in the specified organization.
            ValueError: If the dataset was not loaded with a creds_key and one is not provided.
        """
        if creds_key is None:
            creds_key = self.dataset_creds_key
            if creds_key is None:
                raise ValueError(
                    "The creds_key argument must be provided as the dataset was not loaded with a creds_key."
                )

            if org_id is None and dest_path is None:
                org_id = self.dataset_creds_key_org_id

            if token is None:
                token = self.dataset_creds_key_token

        try:
            path = connect_dataset_entry(
                src_path=self.path,
                dest_path=dest_path,
                org_id=org_id,
                ds_name=ds_name,
                creds_key=creds_key,
                token=token,
            )
        except BadRequestException:
            check_param = "organization id" if org_id else "dataset path"
            raise TokenPermissionError(
                "You do not have permission to create a dataset in the specified "
                + check_param
                + "."
                + " Please check the "
                + check_param
                + " and make sure"
                + "that you have sufficient permissions to the organization."
            )
        self.__class__ = (
            deeplake.core.dataset.deeplake_cloud_dataset.DeepLakeCloudDataset
        )
        self._token = token
        self.path = path
        self.public = False
        self._load_link_creds()
        self._first_load_init(verbose=False)

        base_storage = get_base_storage(self.storage)
        if base_storage is not None and isinstance(base_storage, S3Provider):
            base_storage.creds_used = "PLATFORM"

    def add_creds_key(self, creds_key: str, managed: bool = False):
        """Adds a new creds key to the dataset. These keys are used for tensors that are linked to external data.

        Examples:

            >>> # create/load a dataset
            >>> ds = deeplake.empty("path/to/dataset")
            >>> # add a new creds key
            >>> ds.add_creds_key("my_s3_key")

        Args:
            creds_key (str): The key to be added.
            managed (bool):
                - If ``True``, the creds corresponding to the key will be fetched from Activeloop platform.
                - Defaults to ``False``.

        Raises:
            ValueError: If the dataset is not connected to Activeloop platform and ``managed`` is ``True``.

        Note:
            ``managed`` parameter is applicable only for datasets that are connected to `Activeloop platform <https://app.activeloop.ai>`_.
        """
        if managed:
            raise ValueError(
                "Managed creds are not supported for datasets that are not connected to activeloop platform."
            )
        self.link_creds.add_creds_key(creds_key)
        save_link_creds(self.link_creds, self.storage)

    def populate_creds(
        self,
        creds_key: str,
        creds: Optional[dict] = None,
        from_environment: bool = False,
    ):
        """Populates the creds key added in add_creds_key with the given creds. These creds are used to fetch the external data.
        This needs to be done everytime the dataset is reloaded for datasets that contain links to external data.

        Examples:

            >>> # create/load a dataset
            >>> ds = deeplake.dataset("path/to/dataset")
            >>> # add a new creds key
            >>> ds.add_creds_key("my_s3_key")
            >>> # populate the creds
            >>> ds.populate_creds("my_s3_key", {"aws_access_key_id": "my_access_key", "aws_secret_access_key": "my_secret_key"})
            >>> # or
            >>> ds.populate_creds("my_s3_key", from_environment=True)

        """
        if creds and from_environment:
            raise ValueError(
                "Only one of creds or from_environment can be provided. Both cannot be provided at the same time."
            )
        if from_environment:
            creds = {}
        self.link_creds.populate_creds(creds_key, creds)

    def update_creds_key(
        self,
        creds_key: str,
        new_creds_key: Optional[str] = None,
        managed: Optional[bool] = None,
    ):
        """Updates the name and/or management status of a creds key.

        Args:
            creds_key (str): The key whose name and/or management status is to be changed.
            new_creds_key (str, optional): The new key to replace the old key. If not provided, the old key will be used.
            managed (bool): The target management status. If ``True``, the creds corresponding to the key will be fetched from activeloop platform.

        Raises:
            ValueError: If the dataset is not connected to activeloop platform.
            ValueError: If both ``new_creds_key`` and ``managed`` are ``None``.
            KeyError: If the creds key is not present in the dataset.

        Examples:

            >>> # create/load a dataset
            >>> ds = deeplake.dataset("path/to/dataset")
            >>> # add a new creds key
            >>> ds.add_creds_key("my_s3_key")
            >>> # Populate the name added with creds dictionary
            >>> # These creds are only present temporarily and will have to be repopulated on every reload
            >>> ds.populate_creds("my_s3_key", {})
            >>> # Rename the key and change the management status of the key to True. Before doing this, ensure that the creds have been created on activeloop platform
            >>> # Now, this key will no longer use the credentials populated in the previous step but will instead fetch them from activeloop platform
            >>> # These creds don't have to be populated again on every reload and will be fetched every time the dataset is loaded
            >>> ds.update_creds_key("my_s3_key", "my_managed_key", True)

        """
        if new_creds_key is None and managed is None:
            raise ValueError(
                "Atleast one of new_creds_key or managed must be provided."
            )
        if managed:
            raise ValueError(
                "Managed creds are not supported for datasets that are not connected to activeloop platform."
            )
        replaced_indices = self.link_creds.replace_creds(creds_key, new_creds_key)
        save_link_creds(
            self.link_creds, self.storage, replaced_indices=replaced_indices
        )

    def get_creds_keys(self) -> Set[str]:
        """Returns the set of creds keys added to the dataset. These are used to fetch external data in linked tensors"""
        return set(self.link_creds.creds_keys)

    def get_managed_creds_keys(self) -> List[str]:
        """Returns the list of creds keys added to the dataset that are managed by Activeloop platform. These are used to fetch external data in linked tensors."""
        raise ValueError(
            "Managed creds are not supported for datasets that are not connected to activeloop platform."
        )

    def visualize(
        self, width: Union[int, str, None] = None, height: Union[int, str, None] = None
    ):
        """
        Visualizes the dataset in the Jupyter notebook.

        Args:
            width: Union[int, str, None] Optional width of the visualizer canvas.
            height: Union[int, str, None] Optional height of the visualizer canvas.

        Raises:
            Exception: If the dataset is not a Deep Lake cloud dataset and the visualization is attempted in colab.
        """
        from deeplake.visualizer import visualize

        deeplake_reporter.feature_report(
            feature_name="visualize", parameters={"width": width, "height": height}
        )

        if is_colab():
            provider = self.storage.next_storage
            if isinstance(provider, S3Provider):
                creds = {
                    "aws_access_key_id": provider.aws_access_key_id,
                    "aws_secret_access_key": provider.aws_secret_access_key,
                    "aws_session_token": provider.aws_session_token,
                    "aws_region": provider.aws_region,
                    "endpoint_url": provider.endpoint_url,
                }
                visualize(
                    provider.path,
                    link_creds=self.link_creds,
                    token=self.token,
                    creds=creds,
                )
            else:
                raise Exception(
                    "Cannot visualize non Deep Lake cloud dataset in Colab."
                )
        else:
            visualize(
                self.storage, link_creds=self.link_creds, width=width, height=height
            )

    def __contains__(self, tensor: str):
        return tensor in self.tensors

    def _optimize_and_copy_view(
        self,
        info,
        path: str,
        tensors: Optional[List[str]] = None,
        external=False,
        unlink=True,
        num_workers=0,
        scheduler="threaded",
        progressbar=True,
    ):
        tql_query = info.get("tql_query")
        vds = self._sub_ds(".queries/" + path, verbose=False)
        view = vds._get_view(not external)
        new_path = path + "_OPTIMIZED"
        if tql_query is not None:
            view = view.query(tql_query)
            view.indra_ds.materialize(new_path, tensors, True)
            optimized = self._sub_ds(".queries/" + new_path, empty=False, verbose=False)
        else:
            optimized = self._sub_ds(".queries/" + new_path, empty=True, verbose=False)
            view._copy(
                optimized,
                tensors=tensors,
                overwrite=True,
                unlink=unlink,
                create_vds_index_tensor=True,
                num_workers=num_workers,
                scheduler=scheduler,
                progressbar=progressbar,
            )
        optimized.info.update(vds.info.__getstate__())
        return (vds, optimized, new_path)

    def _optimize_saved_view(
        self,
        id: str,
        tensors: Optional[List[str]] = None,
        external=False,
        unlink=True,
        num_workers=0,
        scheduler="threaded",
        progressbar=True,
    ):
        try:
            with self._temp_write_access():
                with self._lock_queries_json():
                    qjson = self._read_queries_json()
                    idx = -1
                    for i in range(len(qjson)):
                        if qjson[i]["id"] == id:
                            idx = i
                            break
                    if idx == -1:
                        raise KeyError(f"View with id {id} not found.")
                    info = qjson[i]
                    if not info["virtual-datasource"]:
                        # Already optimized
                        return info
                    path = info.get("path", info["id"])
                    old, new, new_path = self._optimize_and_copy_view(
                        info,
                        path,
                        tensors=tensors,
                        unlink=unlink,
                        num_workers=num_workers,
                        scheduler=scheduler,
                        progressbar=progressbar,
                    )
                    new.info["virtual-datasource"] = False
                    new.info["path"] = new_path
                    new.flush()
                    info["virtual-datasource"] = False
                    info["path"] = new_path
                    self._write_queries_json(qjson)
                old.base_storage.disable_readonly()
                try:
                    old.base_storage.clear()
                except Exception as e:
                    warnings.warn(
                        f"Error while deleting old view after writing optimized version: {e}"
                    )
                return info
        except ReadOnlyModeError as e:
            raise ReadOnlyModeError(
                f"You do not have permission to materialize views in this dataset ({self.path})."
            ) from e

    def _sample_indices(self, maxlen: int):
        vds_index = self._tensors(include_hidden=True).get("VDS_INDEX")
        if vds_index:
            return vds_index.numpy().reshape(-1).tolist()
        return self.index.values[0].indices(maxlen)

    @property
    def sample_indices(self):
        """Returns all the indices pointed to by this dataset view."""
        return self._sample_indices(min(t.num_samples for t in self.tensors.values()))

    def _enable_padding(self):
        self._pad_tensors = True

    def _disable_padding(self):
        self._pad_tensors = False

    def _pop(self, index: List[int]):
        """Removes elements at the given indices."""
        with self:
            for tensor in self.tensors.values():
                tensor._pop(index)

    @invalid_view_op
    def pop(self, index: Optional[int] = None):
        """
        Removes a sample from all the tensors of the dataset.
        For any tensor if the index >= len(tensor), the sample won't be popped from it.

        Args:
            index (int, Optional): The index of the sample to be removed. If it is ``None``, the index becomes the ``length of the longest tensor - 1``.

        Raises:
            ValueError: If duplicate indices are provided.
            IndexError: If the index is out of range.
        """
        if index is None:
            index = [self.max_len - 1]

        if not isinstance(index, list):
            index = [index]

        if not index:
            return

        if len(set(index)) != len(index):
            raise ValueError("Duplicate indices are not allowed.")

        max_len = self.max_len
        if max_len == 0:
            raise IndexError("Can't pop from empty dataset.")

        for idx in index:
            if idx < 0:
                raise IndexError("Pop doesn't support negative indices.")
            elif idx >= max_len:
                raise IndexError(
                    f"Index {idx} is out of range. The longest tensor has {max_len} samples."
                )

        index = sorted(index, reverse=True)

        self._pop(index)
        row_ids = index[:]

        index_maintenance.index_operation_dataset(
            self,
            dml_type=_INDEX_OPERATION_MAPPING["REMOVE"],
            rowids=row_ids,
        )

    @property
    def is_view(self) -> bool:
        """Returns ``True`` if this dataset is a view and ``False`` otherwise."""
        return (
            not self.index.is_trivial()
            or hasattr(self, "_vds")
            or hasattr(self, "_view_entry")
        )

    @property
    def is_optimized(self) -> bool:
        return not getattr(getattr(self, "_view_entry", None), "virtual", True)

    @property
    def min_view(self):
        """Returns a view of the dataset in which all tensors are sliced to have the same length as
        the shortest tensor.

        Example:

            Creating a dataset with 5 images and 4 labels. ``ds.min_view`` will return a view in which tensors are
            sliced to have 4 samples.

            >>> import deeplake
            >>> ds = deeplake.dataset("../test/test_ds", overwrite=True)
            >>> ds.create_tensor("images", htype="link[image]", sample_compression="jpg")
            >>> ds.create_tensor("labels", htype="class_label")
            >>> ds.images.extend([deeplake.link("https://picsum.photos/20/20") for _ in range(5)])
            >>> ds.labels.extend([0, 1, 2, 1])
            >>> len(ds.images)
            5
            >>> len(ds.labels)
            4
            >>> for i, sample in enumerate(ds.max_view):
            ...     print(sample["images"].shape, sample["labels"].numpy())
            ...
            (20, 20, 3) [0]
            (20, 20, 3) [1]
            (20, 20, 3) [2]
            (20, 20, 3) [1]

        """
        min_length = min(map(len, self.tensors.values()))
        return self[:min_length]

    @property
    def max_view(self):
        """Returns a view of the dataset in which shorter tensors are padded with ``None`` s to have the same length as
        the longest tensor.

        Example:

            Creating a dataset with 5 images and 4 labels. ``ds.max_view`` will return a view with ``labels`` tensor
            padded to have 5 samples.

            >>> import deeplake
            >>> ds = deeplake.dataset("../test/test_ds", overwrite=True)
            >>> ds.create_tensor("images", htype="link[image]", sample_compression="jpg")
            >>> ds.create_tensor("labels", htype="class_label")
            >>> ds.images.extend([deeplake.link("https://picsum.photos/20/20") for _ in range(5)])
            >>> ds.labels.extend([0, 1, 2, 1])
            >>> len(ds.images)
            5
            >>> len(ds.labels)
            4
            >>> for i, sample in enumerate(ds.max_view):
            ...     print(sample["images"].shape, sample["labels"].numpy())
            ...
            (20, 20, 3) [0]
            (20, 20, 3) [1]
            (20, 20, 3) [2]
            (20, 20, 3) [1]
            (20, 20, 3) [None]
        """
        return self.__class__(
            storage=self.storage,
            index=self.index,
            group_index=self.group_index,
            read_only=self.read_only,
            token=self._token,
            verbose=False,
            version_state=self.version_state,
            path=self.path,
            link_creds=self.link_creds,
            pad_tensors=True,
            enabled_tensors=self.enabled_tensors,
            libdeeplake_dataset=self.libdeeplake_dataset,
            index_params=self.index_params,
        )

    def random_split(self, lengths: Sequence[Union[int, float]]):
        """Splits the dataset into non-overlapping :class:`~deeplake.core.dataset.Dataset` objects of given lengths.
        If a list of fractions that sum up to 1 is given, the lengths will be computed automatically as floor(frac * len(dataset)) for each fraction provided.

        After computing the lengths, if there are any remainders, 1 count will be distributed in round-robin fashion to the lengths until there are no remainders left.

        Example:

            >>> import deeplake
            >>> ds = deeplake.dataset("../test/test_ds", overwrite=True)
            >>> ds.create_tensor("labels", htype="class_label")
            >>> ds.labels.extend([0, 1, 2, 1, 3])
            >>> len(ds)
            5
            >>> train_ds, val_ds = ds.random_split([0.8, 0.2])
            >>> len(train_ds)
            4
            >>> len(val_ds)
            1
            >>> train_ds, val_ds = ds.random_split([3, 2])
            >>> len(train_ds)
            3
            >>> len(val_ds)
            2
            >> train_loader = train_ds.pytorch(batch_size=2, shuffle=True)
            >> val_loader = val_ds.pytorch(batch_size=2, shuffle=False)

        Args:
            lengths (Sequence[Union[int, float]]): lengths or fractions of splits to be produced.

        Returns:
            Tuple[Dataset, ...]: a tuple of datasets of the given lengths.

        Raises:
            ValueError: If the sum of the lengths is not equal to the length of the dataset.
            ValueError: If the dataset has variable length tensors.
            ValueError: If lengths are floats and one or more of them are not between 0 and 1.

        """
        if self.max_len != self.min_len:
            raise ValueError(
                "Random_split is not supported for datasets with variable length tensors."
            )
        return create_random_split_views(self, lengths)

    def _temp_write_access(self):
        # Defined in DeepLakeCloudDataset
        return memoryview(b"")  # No-op context manager

    def _get_storage_repository(self) -> Optional[str]:
        return getattr(self.base_storage, "repository", None)
