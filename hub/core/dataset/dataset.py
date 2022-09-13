# type: ignore
import os
import uuid
import sys
import json
import posixpath
from logging import warning
from collections import Iterable
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from functools import partial
from itertools import chain

import jwt
import pathlib
import numpy as np
from time import time
from tqdm import tqdm  # type: ignore

import hub
from hub.core.index.index import IndexEntry
from hub.core.link_creds import LinkCreds
from hub.util.invalid_view_op import invalid_view_op
from hub.api.info import load_info
from hub.client.log import logger
from hub.client.utils import get_user_name
from hub.constants import (
    FIRST_COMMIT_ID,
    DEFAULT_MEMORY_CACHE_SIZE,
    DEFAULT_LOCAL_CACHE_SIZE,
    MB,
    SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE,
    DEFAULT_READONLY,
)
from hub.core.fast_forwarding import ffw_dataset_meta
from hub.core.index import Index
from hub.core.lock import lock_dataset, unlock_dataset, Lock
from hub.core.meta.dataset_meta import DatasetMeta
from hub.core.storage import (
    LRUCache,
    S3Provider,
    GCSProvider,
    MemoryProvider,
)
from hub.core.tensor import Tensor, create_tensor, delete_tensor
from hub.core.version_control.commit_node import CommitNode  # type: ignore
from hub.core.version_control.dataset_diff import load_dataset_diff
from hub.htype import (
    HTYPE_CONFIGURATIONS,
    UNSPECIFIED,
    verify_htype_key_value,
)
from hub.integrations import dataset_to_tensorflow
from hub.util.bugout_reporter import hub_reporter, feature_report_path
from hub.util.dataset import try_flushing
from hub.util.cache_chain import generate_chain
from hub.util.hash import hash_inputs
from hub.util.htype import parse_complex_htype
from hub.util.link import save_link_creds
from hub.util.merge import merge
from hub.util.notebook import is_colab
from hub.util.path import convert_pathlib_to_string_if_needed, get_org_id_and_ds_name
from hub.util.logging import log_visualizer_link
from hub.util.warnings import always_warn
from hub.util.exceptions import (
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
    NotLoggedInError,
    RenameError,
    EmptyCommitError,
    DatasetViewSavingError,
    DatasetHandlerError,
    EmptyTensorError,
    SampleAppendingError,
)
from hub.util.keys import (
    dataset_exists,
    get_dataset_info_key,
    get_dataset_meta_key,
    tensor_exists,
    get_queries_key,
    get_queries_lock_key,
    get_sample_id_tensor_key,
    get_sample_info_tensor_key,
    get_sample_shape_tensor_key,
    filter_name,
    get_tensor_meta_key,
    get_tensor_commit_diff_key,
    get_tensor_tile_encoder_key,
    get_tensor_info_key,
    get_tensor_commit_chunk_set_key,
    get_chunk_id_encoder_key,
    get_dataset_diff_key,
    get_sequence_encoder_key,
    get_dataset_linked_creds_key,
)
from hub.util.path import get_path_from_storage
from hub.util.remove_cache import get_base_storage
from hub.util.diff import get_all_changes_string, get_changes_and_messages
from hub.util.version_control import (
    auto_checkout,
    checkout,
    commit,
    current_commit_has_change,
    load_meta,
    warn_node_checkout,
    load_version_info,
    copy_metas,
    create_commit_chunk_sets,
)
from hub.util.pretty_print import summary_dataset
from hub.core.dataset.view_entry import ViewEntry
from hub.hooks import dataset_read
from itertools import chain
import warnings
import jwt


_LOCKABLE_STORAGES = {S3Provider, GCSProvider}


class Dataset:
    def __init__(
        self,
        storage: LRUCache,
        index: Optional[Index] = None,
        group_index: str = "",
        read_only: Optional[bool] = None,
        public: Optional[bool] = False,
        token: Optional[str] = None,
        verbose: bool = True,
        version_state: Optional[Dict[str, Any]] = None,
        path: Optional[Union[str, pathlib.Path]] = None,
        is_iteration: bool = False,
        link_creds=None,
        pad_tensors: bool = False,
        lock: bool = True,
        **kwargs,
    ):
        """Initializes a new or existing dataset.

        Args:
            storage (LRUCache): The storage provider used to access the dataset.
            index (Index, Optional): The Index object restricting the view of this dataset's tensors.
            group_index (str): Name of the group this dataset instance represents.
            read_only (bool, Optional): Opens dataset in read only mode if this is passed as True. Defaults to False.
                Datasets stored on Hub cloud that your account does not have write access to will automatically open in read mode.
            public (bool, Optional): Applied only if storage is Hub cloud storage and a new Dataset is being created. Defines if the dataset will have public access.
            token (str, Optional): Activeloop token, used for fetching credentials for Hub datasets. This is Optional, tokens are normally autogenerated.
            verbose (bool): If ``True``, logs will be printed. Defaults to True.
            version_state (Dict[str, Any], Optional): The version state of the dataset, includes commit_id, commit_node, branch, branch_commit_map and commit_node_map.
            path (str, pathlib.Path): The path to the dataset.
            is_iteration (bool): If this Dataset is being used as an iterator.
            link_creds (LinkCreds, Optional): The LinkCreds object used to access tensors that have external data linked to them.
            pad_tensors (bool): If ``True``, shorter tensors will be padded to the length of the longest tensor.
            **kwargs: Passing subclass variables through without errors.
            lock (bool): Whether the dataset should be locked for writing. Only applicable for s3, hub and gcs datasets. No effect if read_only=True.


        Raises:
            ValueError: If an existing local path is given, it must be a directory.
            ImproperDatasetInitialization: Exactly one argument out of 'path' and 'storage' needs to be specified.
                This is raised if none of them are specified or more than one are specifed.
            InvalidHubPathException: If a Hub cloud path (path starting with `hub://`) is specified and it isn't of the form `hub://username/datasetname`.
            AuthorizationException: If a Hub cloud path (path starting with `hub://`) is specified and the user doesn't have access to the dataset.
            PathNotEmptyException: If the path to the dataset doesn't contain a Hub dataset and is also not empty.
            LockedException: If read_only is False but the dataset is locked for writing by another machine.
            ReadOnlyModeError: If read_only is False but write access is not available.
        """
        d: Dict[str, Any] = {}
        d["_client"] = d["org_id"] = d["ds_name"] = None
        # uniquely identifies dataset
        d["path"] = convert_pathlib_to_string_if_needed(path) or get_path_from_storage(
            storage
        )
        d["storage"] = storage
        d["_read_only_error"] = read_only is False
        d["_read_only"] = DEFAULT_READONLY if read_only is None else read_only
        d["base_storage"] = get_base_storage(storage)
        d["_locked_out"] = False  # User requested write access but was denied
        d["is_iteration"] = is_iteration
        d["is_first_load"] = version_state is None
        d["_is_filtered_view"] = False
        d["index"] = index or Index()
        d["group_index"] = group_index
        d["_token"] = token
        d["public"] = public
        d["verbose"] = verbose
        d["version_state"] = version_state or {}
        d["link_creds"] = link_creds
        d["_info"] = None
        d["_ds_diff"] = None
        d["_view_id"] = str(uuid.uuid4)
        d["_view_invalid"] = False
        d["_waiting_for_view_base_commit"] = False
        d["_new_view_base_commit"] = None
        d["_view_base"] = None
        d["_update_hooks"] = {}
        d["_commit_hooks"] = {}
        d["_parent_dataset"] = None
        d["_pad_tensors"] = pad_tensors
        d["_locking_enabled"] = lock

        self.__dict__.update(d)
        try:
            self._set_derived_attributes()
        except LockedException:
            raise LockedException(
                "This dataset cannot be open for writing as it is locked by another machine. Try loading the dataset with `read_only=True`."
            )
        except ReadOnlyModeError as e:
            raise ReadOnlyModeError(
                "This dataset cannot be open for writing as you don't have permissions. Try loading the dataset with `read_only=True."
            )
        self._first_load_init()
        self._initial_autoflush: List[
            bool
        ] = []  # This is a stack to support nested with contexts

    def _lock_lost_handler(self):
        """This is called when lock is acquired but lost later on due to slow update."""
        self.read_only = True
        if self.verbose:
            always_warn(
                "Unable to update dataset lock as another machine has locked it for writing. Switching to read only mode."
            )
        self._locked_out = True

    def __enter__(self):
        self._initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.storage.autoflush = self._initial_autoflush.pop()
        if not self._read_only:
            self.storage.maybe_flush()

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

    def __len__(self):
        """Returns the length of the smallest tensor"""
        tensor_lengths = [len(tensor) for tensor in self.tensors.values()]
        if min(tensor_lengths, default=0) != max(tensor_lengths, default=0):
            warning(
                "The length of tensors in the dataset is different. The len(ds) returns the length of the "
                "smallest tensor in the dataset. If you want the length of the longest tensor in the dataset use "
                "ds.max_len."
            )
        length_fn = max if self._pad_tensors else min
        return length_fn(tensor_lengths, default=0)

    @property
    def max_len(self):
        """Return the maximum length of the tensor"""
        return max([len(tensor) for tensor in self.tensors.values()])

    @property
    def min_len(self):
        """Return the minimum length of the tensor"""
        return min([len(tensor) for tensor in self.tensors.values()])

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
            "_view_invalid",
            "_new_view_base_commit",
            "_parent_dataset",
            "_pad_tensors",
            "_locking_enabled",
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
        state["is_iteration"] = False
        state["_read_only_error"] = False
        state["_initial_autoflush"] = []
        state["_ds_diff"] = None
        state["_view_base"] = None
        state["_update_hooks"] = {}
        state["_commit_hooks"] = {}
        state["_waiting_for_view_base_commit"] = False
        state["_client"] = state["org_id"] = state["ds_name"] = None
        self.__dict__.update(state)
        self.__dict__["base_storage"] = get_base_storage(self.storage)
        # clear cache while restoring
        self.storage.clear_cache_without_flush()
        self._set_derived_attributes(verbose=False)

    def __getitem__(
        self,
        item: Union[
            str, int, slice, List[int], Tuple[Union[int, slice, Tuple[int]]], Index
        ],
        is_iteration: bool = False,
    ):
        if isinstance(item, str):
            fullpath = posixpath.join(self.group_index, item)
            tensor = self._get_tensor_from_root(fullpath)
            if tensor is not None:
                return tensor[self.index]
            elif self._has_group_in_root(fullpath):
                ret = self.__class__(
                    storage=self.storage,
                    index=self.index,
                    group_index=posixpath.join(self.group_index, item),
                    read_only=self.read_only,
                    token=self._token,
                    verbose=False,
                    version_state=self.version_state,
                    path=self.path,
                    link_creds=self.link_creds,
                    pad_tensors=self._pad_tensors,
                )
            elif "/" in item:
                splt = posixpath.split(item)
                ret = self[splt[0]][splt[1]]
            else:
                raise TensorDoesNotExistError(item)
        elif isinstance(item, (int, slice, list, tuple, Index)):
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
            )
        else:
            raise InvalidKeyTypeError(item)
        ret._view_base = self._view_base or self
        if hasattr(self, "_view_entry"):
            ret._view_entry = self._view_entry
        return ret

    @invalid_view_op
    @hub_reporter.record_call
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
        verify: bool = False,
        exist_ok: bool = False,
        **kwargs,
    ):
        """Creates a new tensor in the dataset.

        Examples:
            >>> # create dataset
            >>> ds = hub.dataset("path/to/dataset")

            >>> # create tensors
            >>> ds.create_tensor("images", htype="image", sample_compression="jpg")
            >>> ds.create_tensor("videos", htype="video", sample_compression="mp4")
            >>> ds.create_tensor("data")
            >>> ds.create_tensor("point_clouds", htype="point_cloud")

            >>> # append data
            >>> ds.images.append(np.ones((400, 400, 3), dtype='uint8'))
            >>> ds.videos.append(hub.read("videos/sample_video.mp4"))
            >>> ds.data.append(np.zeros((100, 100, 2)))

        Args:
            name (str): The name of the tensor to be created.
            htype (str):
                - The class of data for the tensor.
                - The defaults for other parameters are determined in terms of this value.
                - For example, ``htype="image"`` would have ``dtype`` default to ``uint8``.
                - These defaults can be overridden by explicitly passing any of the other parameters to this function.
                - May also modify the defaults for other parameters.
            dtype (str): Optionally override this tensor's ``dtype``. All subsequent samples are required to have this ``dtype``.
            sample_compression (str): All samples will be compressed in the provided format. If ``None``, samples are uncompressed.
            chunk_compression (str): All chunks will be compressed in the provided format. If ``None``, chunks are uncompressed.
            hidden (bool): If ``True``, the tensor will be hidden from ds.tensors but can still be accessed via ``ds[tensor_name]``.
            create_sample_info_tensor (bool): If ``True``, meta data of individual samples will be saved in a hidden tensor. This data can be accessed via :attr:`tensor[i].sample_info <hub.core.tensor.Tensor.sample_info>`.
            create_shape_tensor (bool): If ``True``, an associated tensor containing shapes of each sample will be created.
            create_id_tensor (bool): If ``True``, an associated tensor containing unique ids for each sample will be created. This is useful for merge operations.
            verify (bool): Valid only for link htypes. If ``True``, all links will be verified before they are added to the tensor.
            exist_ok (bool): If ``True``, the group is created if it does not exist. if ``False``, an error is raised if the group already exists.
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
                "sample_compression": sample_compression,
                "chunk_compression": chunk_compression,
                "hidden": hidden,
                "is_link": is_link,
                "is_sequence": is_sequence,
            }
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

        kwargs["is_sequence"] = kwargs.get("is_sequence") or is_sequence
        kwargs["is_link"] = kwargs.get("is_link") or is_link
        kwargs["verify"] = verify
        if is_link and (
            sample_compression != UNSPECIFIED or chunk_compression != UNSPECIFIED
        ):
            warnings.warn(
                "Chunk_compression and sample_compression aren't valid for tensors with linked data. Ignoring these arguments."
            )
            sample_compression = UNSPECIFIED
            chunk_compression = UNSPECIFIED

        if not self._is_root():
            return self.root.create_tensor(
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
        ):
            self._create_sample_info_tensor(name)
        if create_shape_tensor and htype not in ("text", "json"):
            self._create_sample_shape_tensor(name, htype=htype)
        if create_id_tensor:
            self._create_sample_id_tensor(name)
        return tensor

    def _create_sample_shape_tensor(self, tensor: str, htype: str):
        shape_tensor = get_sample_shape_tensor_key(tensor)
        self.create_tensor(
            shape_tensor,
            hidden=True,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            max_chunk_size=SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE,
        )
        f = "append_len" if htype == "list" else "append_shape"
        self._link_tensors(
            tensor, shape_tensor, append_f=f, update_f=f, flatten_sequence=True
        )

    def _create_sample_id_tensor(self, tensor: str):
        id_tensor = get_sample_id_tensor_key(tensor)
        self.create_tensor(
            id_tensor,
            hidden=True,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
        )
        self._link_tensors(
            tensor,
            id_tensor,
            append_f="append_id",
            flatten_sequence=False,
        )

    def _create_sample_info_tensor(self, tensor: str):
        sample_info_tensor = get_sample_info_tensor_key(tensor)
        self.create_tensor(
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
            "append_info",
            "update_info",
            flatten_sequence=True,
        )

    def _hide_tensor(self, tensor: str):
        self._tensors()[tensor].meta.set_hidden(True)
        self.meta._hide_tensor(tensor)
        self.storage.maybe_flush()

    @invalid_view_op
    @hub_reporter.record_call
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
        """
        auto_checkout(self)

        name = filter_name(name, self.group_index)
        key = self.version_state["tensor_names"].get(name)

        if not key:
            raise TensorDoesNotExistError(name)

        if not tensor_exists(key, self.storage, self.version_state["commit_id"]):
            raise TensorDoesNotExistError(name)

        if not self._is_root():
            return self.root.delete_tensor(name, large_ok)

        if not large_ok:
            chunk_engine = self.version_state["full_tensors"][key].chunk_engine
            size_approx = chunk_engine.num_samples * chunk_engine.min_chunk_size
            if size_approx > hub.constants.DELETE_SAFETY_SIZE:
                logger.info(
                    f"Tensor {name} was too large to delete. Try again with large_ok=True."
                )
                return

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
                self.delete_tensor(t_name, large_ok=True)

        self.storage.flush()

    @invalid_view_op
    @hub_reporter.record_call
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
        auto_checkout(self)

        full_path = filter_name(name, self.group_index)

        if full_path not in self._groups:
            raise TensorGroupDoesNotExistError(name)

        if not self._is_root():
            return self.root.delete_group(full_path, large_ok)

        if not large_ok:
            size_approx = self[name].size_approx()
            if size_approx > hub.constants.DELETE_SAFETY_SIZE:
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
    @hub_reporter.record_call
    def create_tensor_like(
        self, name: str, source: "Tensor", unlink: bool = False
    ) -> "Tensor":
        """Copies the ``source`` tensor's meta information and creates a new tensor with it. No samples are copied, only the meta/info for the tensor is.

        Examples:
            >>> ds.create_tensor_like("cats", ds["images"])

        Args:
            name (str): Name for the new tensor.
            source (Tensor): Tensor who's meta/info will be copied. May or may not be contained in the same dataset.
            unlink (bool): Whether to unlink linked tensors.

        Returns:
            Tensor: New Tensor object.
        """

        info = source.info.__getstate__().copy()
        meta = source.meta.__getstate__().copy()
        if unlink:
            meta["is_link"] = False
        del meta["min_shape"]
        del meta["max_shape"]
        del meta["length"]
        del meta["version"]
        del meta["name"]

        destination_tensor = self.create_tensor(name, **meta)
        destination_tensor.info.update(info)
        return destination_tensor

    def _rename_tensor(self, name, new_name):
        tensor = self[name]
        tensor.meta.name = new_name
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

    @hub_reporter.record_call
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

    @hub_reporter.record_call
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
                posixpath.join(new_name, posixpath.relpath(tensor, name)),
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
        if isinstance(value, (np.ndarray, np.generic)):
            raise TypeError(
                "Setting tensor attributes directly is not supported. To add a tensor, use the `create_tensor` method."
                + "To add data to a tensor, use the `append` and `extend` methods."
            )
        else:
            return super().__setattr__(name, value)

    def __iter__(self):
        dataset_read(self)
        for i in range(len(self)):
            yield self.__getitem__(i, is_iteration=True)

    def _load_version_info(self):
        """Loads data from version_control_file otherwise assume it doesn't exist and load all empty"""
        if self.version_state:
            return

        branch = "main"
        version_state = {"branch": branch}
        try:
            version_info = load_version_info(self.storage)
            version_state["branch_commit_map"] = version_info["branch_commit_map"]
            version_state["commit_node_map"] = version_info["commit_node_map"]
            commit_id = version_state["branch_commit_map"][branch]
            version_state["commit_id"] = commit_id
            version_state["commit_node"] = version_state["commit_node_map"][commit_id]
        except Exception:
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

    def _lock(self, err=False, verbose=True):
        if not self._locking_enabled:
            return True
        storage = self.base_storage
        if storage.read_only and not self._locked_out:
            if err:
                raise ReadOnlyModeError()
            return False

        if isinstance(storage, tuple(_LOCKABLE_STORAGES)) and (
            not self.read_only or self._locked_out
        ):
            try:
                # temporarily disable read only on base storage, to try to acquire lock, if exception, it will be again made readonly
                storage.disable_readonly()
                lock_dataset(
                    self,
                    lock_lost_callback=self._lock_lost_handler,
                )
            except LockedException as e:
                self.read_only = True
                self.__dict__["_locked_out"] = True
                if err:
                    raise e
                if verbose:
                    always_warn(
                        "Checking out dataset in read only mode as another machine has locked this version for writing."
                    )
                return False
        return True

    def _unlock(self):
        unlock_dataset(self)

    def __del__(self):
        try:
            self._unlock()
        except Exception:  # python shutting down
            pass

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
        if not allow_empty and not self.has_head_changes:
            raise EmptyCommitError(
                "There are no changes, commit is not done. Try again with allow_empty=True."
            )

        return self._commit(message)

    @hub_reporter.record_call
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
        if self._is_filtered_view:
            raise Exception(
                "Cannot perform version control operations on a filtered dataset view."
            )

        if conflict_resolution not in [None, "ours", "theirs"]:
            raise ValueError(
                f"conflict_resolution must be one of None, 'ours', or 'theirs'. Got {conflict_resolution}"
            )

        try_flushing(self)

        self._initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False

        merge(self, target_id, conflict_resolution, delete_removed_tensors, force)

        self.storage.autoflush = self._initial_autoflush.pop()

    def _commit(self, message: Optional[str] = None, hash: Optional[str] = None) -> str:
        if self._is_filtered_view:
            raise Exception(
                "Cannot perform version control operations on a filtered dataset view."
            )

        try_flushing(self)

        self._initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False
        try:
            self._unlock()
            commit(self, message, hash)
            self._lock()
        finally:
            self.storage.autoflush = self._initial_autoflush.pop()
        self._info = None
        self._ds_diff = None
        [f() for f in list(self._commit_hooks.values())]
        # do not store commit message
        hub_reporter.feature_report(feature_name="commit", parameters={})

        return self.commit_id  # type: ignore

    def checkout(self, address: str, create: bool = False) -> Optional[str]:
        """Checks out to a specific commit_id or branch. If ``create = True``, creates a new branch with name ``address``.

        Args:
            address (str): The commit_id or branch to checkout to.
            create (bool): If ``True``, creates a new branch with name as address.

        Returns:
            Optional[str]: The commit_id of the dataset after checkout.

        Raises:
            Exception: If the dataset is a filtered view.

        Examples:

            >>> ds = hub.empty("../test/test_ds")
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
        return self._checkout(address, create)

    def _checkout(
        self,
        address: str,
        create: bool = False,
        hash: Optional[str] = None,
        verbose=True,
    ) -> Optional[str]:
        if self._is_filtered_view:
            raise Exception(
                "Cannot perform version control operations on a filtered dataset view."
            )
        if self._locked_out:
            self.storage.disable_readonly()
            self._read_only = False
            self.base_storage.disable_readonly()
        try_flushing(self)
        self._initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False
        err = False
        try:
            self._unlock()
            checkout(self, address, create, hash)
        except Exception as e:
            err = True
            if self._locked_out:
                self.storage.enable_readonly()
                self._read_only = True
                self.base_storage.enable_readonly()
            raise e
        finally:
            if not (err and self._locked_out):
                self._lock(verbose=verbose)
            self.storage.autoflush = self._initial_autoflush.pop()
        self._info = None
        self._ds_diff = None

        # do not store address
        hub_reporter.feature_report(
            feature_name="checkout", parameters={"Create": str(create)}
        )
        commit_node = self.version_state["commit_node"]
        if self.verbose:
            warn_node_checkout(commit_node, create)

        return self.commit_id

    @hub_reporter.record_call
    def log(self):
        """Displays the details of all the past commits."""
        commit_node = self.version_state["commit_node"]
        print("---------------\nHub Version Log\n---------------\n")
        print(f"Current Branch: {self.version_state['branch']}")
        if self.has_head_changes:
            print("** There are uncommitted changes on this branch.")
        print()
        while commit_node:
            if not commit_node.is_head_node:
                print(f"{commit_node}\n")
            commit_node = commit_node.parent

    @hub_reporter.record_call
    def diff(
        self, id_1: Optional[str] = None, id_2: Optional[str] = None, as_dict=False
    ) -> Optional[Dict]:
        """Returns/displays the differences between commits/branches.

        For each tensor this contains information about the sample indexes that were added/modified as well as whether the tensor was created.

        Args:
            id_1 (str, Optional): The first commit_id or branch name.
            id_2 (str, Optional): The second commit_id or branch name.
            as_dict (bool, Optional): If ``True``, returns a dictionary of the differences instead of printing them.
                This dictionary will have two keys - "tensor" and "dataset" which represents tensor level and dataset level changes, respectively.
                Defaults to False.

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

                - If ``id_1`` and ``id_2`` are None, a dictionary containing the differences between the current state and the previous commit will be returned.
                - If only ``id_1`` is provided, a dictionary containing the differences in the current state and ``id_1`` respectively will be returned.
                - If only ``id_2`` is provided, a ValueError will be raised.
                - If both ``id_1`` and ``id_2`` are provided, a dictionary containing the differences in ``id_1`` and ``id_2`` respectively will be returned.

            ``None`` is returned if ``as_dict`` is ``False``.

            Example of a dict returned:

            >>> {
            ...    "image": {"data_added": [3, 6], "data_updated": {0, 2}, "created": False, "info_updated": False, "data_transformed_in_place": False},
            ...    "label": {"data_added": [0, 3], "data_updated": {}, "created": True, "info_updated": False, "data_transformed_in_place": False},
            ...    "other/stuff" : {data_added: [3, 3], data_updated: {1, 2}, created: True, "info_updated": False, "data_transformed_in_place": False},
            ... }


            - Here, "data_added" is a range of sample indexes that were added to the tensor.

                - For example [3, 6] means that sample 3, 4 and 5 were added.
                - Another example [3, 3] means that no samples were added as the range is empty.

            - "data_updated" is a set of sample indexes that were updated.

                - For example {0, 2} means that sample 0 and 2 were updated.

            - "created" is a boolean that is ``True`` if the tensor was created.

            - "info_updated" is a boolean that is ``True`` if the info of the tensor was updated.

            - "data_transformed_in_place" is a boolean that is ``True`` if the data of the tensor was transformed in place.
        """
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

    def _populate_meta(self, verbose=True):
        """Populates the meta information for the dataset."""
        if dataset_exists(self.storage):
            if verbose and self.verbose:
                logger.info(f"{self.path} loaded successfully.")
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
            self.storage.register_hub_object(key, meta)
            self._register_dataset()
            self.flush()

    def _register_dataset(self):
        """overridden in HubCloudDataset"""

    def _send_query_progress(self, *args, **kwargs):
        """overridden in HubCloudDataset"""

    def _send_compute_progress(self, *args, **kwargs):
        """overridden in HubCloudDataset"""

    def _send_pytorch_progress(self, *args, **kwargs):
        """overridden in HubCloudDataset"""

    def _send_filter_progress(self, *args, **kwargs):
        """overridden in HubCloudDataset"""

    def _send_commit_event(self, *args, **kwargs):
        """overridden in HubCloudDataset"""

    def _send_dataset_creation_event(self, *args, **kwargs):
        """overridden in HubCloudDataset"""

    def _send_branch_creation_event(self, *args, **kwargs):
        """overridden in HubCloudDataset"""

    def _first_load_init(self):
        """overridden in HubCloudDataset"""

    @property
    def read_only(self):
        return self._read_only

    @property
    def has_head_changes(self):
        """Returns True if currently at head node and uncommitted changes are present."""
        commit_node = self.version_state["commit_node"]
        return not commit_node.children and current_commit_has_change(
            self.version_state, self.storage
        )

    def _set_read_only(self, value: bool, err: bool):
        storage = self.storage
        self.__dict__["_read_only"] = value
        if value:
            storage.enable_readonly()
            if isinstance(storage, LRUCache) and storage.next_storage is not None:
                storage.next_storage.enable_readonly()
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
                raise e

    @read_only.setter
    @invalid_view_op
    def read_only(self, value: bool):
        self._set_read_only(value, True)

    @hub_reporter.record_call
    def pytorch(
        self,
        transform: Optional[Callable] = None,
        tensors: Optional[Sequence[str]] = None,
        tobytes: Union[bool, Sequence[str]] = False,
        num_workers: int = 1,
        batch_size: int = 1,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        shuffle: bool = False,
        buffer_size: int = 2048,
        use_local_cache: bool = False,
        use_progress_bar: bool = False,
        return_index: bool = True,
        pad_tensors: bool = False,
    ):
        """Converts the dataset into a pytorch Dataloader.

        Args:
            transform (Callable, Optional): Transformation function to be applied to each sample.
            tensors (List, Optional): Optionally provide a list of tensor names in the ordering that your training script expects. For example, if you have a dataset that has "image" and "label" tensors, if `tensors=["image", "label"]`, your training script should expect each batch will be provided as a tuple of (image, label).
            tobytes (bool): If ``True``, samples will not be decompressed and their raw bytes will be returned instead of numpy arrays. Can also be a list of tensors, in which case those tensors alone will not be decompressed.
            num_workers (int): The number of workers to use for fetching data in parallel.
            batch_size (int): Number of samples per batch to load. Default value is 1.
            drop_last (bool): Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
                if ``False`` and the size of dataset is not divisible by the batch size, then the last batch will be smaller. Default value is False.
                Read torch.utils.data.DataLoader docs for more details.
            collate_fn (Callable, Optional): merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
                Read torch.utils.data.DataLoader docs for more details.
            pin_memory (bool): If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them. Default value is False.
                Read torch.utils.data.DataLoader docs for more details.
            shuffle (bool): If ``True``, the data loader will shuffle the data indices. Default value is False. Details about how hub shuffles data can be found at https://docs.activeloop.ai/how-hub-works/shuffling-in-ds.pytorch
            buffer_size (int): The size of the buffer used to shuffle the data in MBs. Defaults to 2048 MB. Increasing the buffer_size will increase the extent of shuffling.
            use_local_cache (bool): If ``True``, the data loader will use a local cache to store data. This is useful when the dataset can fit on the machine and we don't want to fetch the data multiple times for each iteration. Default value is False.
            use_progress_bar (bool): If ``True``, tqdm will be wrapped around the returned dataloader. Default value is True.
            return_index (bool): If ``True``, the returned dataloader will have a key "index" that contains the index of the sample(s) in the original dataset. Default value is True.
            pad_tensors (bool): If ``True``, shorter tensors will be padded to the length of the longest tensor. Default value is False.

        Returns:
            A torch.utils.data.DataLoader object.

        Raises:
            EmptyTensorError: If one or more tensors being passed to pytorch are empty.

        Note:
            Pytorch does not support uint16, uint32, uint64 dtypes. These are implicitly type casted to int32, int64 and int64 respectively.
            This spins up it's own workers to fetch data.
        """
        from hub.integrations import dataset_to_pytorch as to_pytorch

        dataloader = to_pytorch(
            self,
            transform=transform,
            tensors=tensors,
            tobytes=tobytes,
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
        )

        if use_progress_bar:
            dataloader = tqdm(dataloader, desc=self.path, total=len(self) // batch_size)
        dataset_read(self)
        return dataloader

    @hub_reporter.record_call
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
                See :class:`hub.core.query.query.DatasetQuery` for more details.
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
            Following filters are identical and return dataset view where all the samples have label equals to 2.

            >>> dataset.filter(lambda sample: sample.labels.numpy() == 2)
            >>> dataset.filter('labels == 2')
        """
        from hub.core.query import filter_dataset, query_dataset

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

    def _get_total_meta(self):
        """Returns tensor metas all together"""
        return {
            tensor_key: tensor_value.meta
            for tensor_key, tensor_value in self.version_state["full_tensors"].items()
        }

    def _set_derived_attributes(self, verbose: bool = True):
        """Sets derived attributes during init and unpickling."""
        if self.is_first_load:
            self.storage.autoflush = True
            self._load_version_info()
            self._load_link_creds()
            self._set_read_only(
                self._read_only, err=self._read_only_error
            )  # TODO: weird fix for dataset unpickling
            self._populate_meta(verbose)  # TODO: use the same scheme as `load_info`
            if self.index.is_trivial():
                self.index = Index.from_json(self.meta.default_index)
        elif not self._read_only:
            self._lock()  # for ref counting

        if not self.is_iteration:
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

    @hub_reporter.record_call
    def tensorflow(
        self,
        tensors: Optional[Sequence[str]] = None,
        tobytes: Union[bool, Sequence[str]] = False,
    ):
        """Converts the dataset into a tensorflow compatible format.

        See https://www.tensorflow.org/api_docs/python/tf/data/Dataset

        Args:
            tensors (List, Optional): Optionally provide a list of tensor names in the ordering that your training script expects. For example, if you have a dataset that has "image" and "label" tensors, if ``tensors=["image", "label"]``, your training script should expect each batch will be provided as a tuple of (image, label).
            tobytes (bool): If ``True``, samples will not be decompressed and their raw bytes will be returned instead of numpy arrays. Can also be a list of tensors, in which case those tensors alone will not be decompressed.

        Returns:
            tf.data.Dataset object that can be used for tensorflow training.
        """
        dataset_read(self)
        return dataset_to_tensorflow(self, tensors=tensors, tobytes=tobytes)

    def flush(self):
        """Necessary operation after writes if caches are being used.
        Writes all the dirty data from the cache layers (if any) to the underlying storage.
        Here dirty data corresponds to data that has been changed/assigned and but hasn't yet been sent to the
        underlying storage.
        """
        self.storage.flush()

    def clear_cache(self):
        """
        - Flushes (see :func:`Dataset.flush`) the contents of the cache layers (if any) and then deletes contents of all the layers of it.
        - This doesn't delete data from the actual storage.
        - This is useful if you have multiple datasets with memory caches open, taking up too much RAM.
        - Also useful when local cache is no longer needed for certain datasets and is taking up storage space.
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
    @hub_reporter.record_call
    def rename(self, path: Union[str, pathlib.Path]):
        """Renames the dataset to `path`.

        Example:

            >>> ds = hub.load("hub://username/dataset")
            >>> ds.rename("hub://username/renamed_dataset")

        Args:
            path (str, pathlib.Path): New path to the dataset.

        Raises:
            RenameError: If ``path`` points to a different directory.
        """
        path = convert_pathlib_to_string_if_needed(path)
        path = path.rstrip("/")
        if posixpath.split(path)[0] != posixpath.split(self.path)[0]:
            raise RenameError
        self.base_storage.rename(path)
        self.path = path

    @invalid_view_op
    @hub_reporter.record_call
    def delete(self, large_ok=False):
        """Deletes the entire dataset from the cache layers (if any) and the underlying storage.
        This is an **IRREVERSIBLE** operation. Data once deleted can not be recovered.

        Args:
            large_ok (bool): Delete datasets larger than 1 GB. Defaults to ``False``.
        """

        if hasattr(self, "_view_entry"):
            self._view_entry.delete()
            return
        if hasattr(self, "_vds"):
            self._vds.delete(large_ok=large_ok)
            return
        if not large_ok:
            size = self.size_approx()
            if size > hub.constants.DELETE_SAFETY_SIZE:
                logger.info(
                    f"Hub Dataset {self.path} was too large to delete. Try again with large_ok=True."
                )
                return

        self._unlock()
        self.storage.clear()

    def summary(self):
        """Prints a summary of the dataset."""
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

        index_str = f"index={self.index}, "
        if self.index.is_trivial():
            index_str = ""

        group_index_str = (
            f"group_index='{self.group_index}', " if self.group_index else ""
        )

        return f"Dataset({path_str}{mode_str}{index_str}{group_index_str}tensors={self._all_tensors_filtered(include_hidden=False)})"

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
        return self._token

    @property
    def _ungrouped_tensors(self) -> Dict[str, Tensor]:
        """Top level tensors in this group that do not belong to any sub groups"""
        return {
            posixpath.basename(k): self.version_state["full_tensors"][v]
            for k, v in self.version_state["tensor_names"].items()
            if posixpath.dirname(k) == self.group_index
        }

    def _all_tensors_filtered(self, include_hidden: bool = True) -> List[str]:
        """Names of all tensors belonging to this group, including those within sub groups"""
        hidden_tensors = self.meta.hidden_tensors
        tensor_names = self.version_state["tensor_names"]
        return [
            posixpath.relpath(t, self.group_index)
            for t in tensor_names
            if (not self.group_index or t.startswith(self.group_index + "/"))
            and (include_hidden or tensor_names[t] not in hidden_tensors)
        ]

    def _tensors(self, include_hidden: bool = True) -> Dict[str, Tensor]:
        """All tensors belonging to this group, including those within sub groups. Always returns the sliced tensors."""
        return {
            t: self.version_state["full_tensors"][
                self.version_state["tensor_names"][posixpath.join(self.group_index, t)]
            ][self.index]
            for t in self._all_tensors_filtered(include_hidden)
        }

    @property
    def tensors(self) -> Dict[str, Tensor]:
        """All tensors belonging to this group, including those within sub groups. Always returns the sliced tensors."""
        return self._tensors(include_hidden=False)

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
        return {
            "commit": commit_node.commit_id,
            "author": commit_node.commit_user_name,
            "time": str(commit_node.commit_time)[:-7],
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
        )
        self.storage.autoflush = autoflush
        return ds

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

    @hub_reporter.record_call
    def create_group(self, name: str, exist_ok=False) -> "Dataset":
        """Creates a tensor group. Intermediate groups in the path are also created.

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
            >>> ds['images'].create_tensor("cats")

            >>> ds.create_groups("images/jpg/cats")
            >>> ds["images"].create_tensor("png")
            >>> ds["images/jpg"].create_group("dogs")
        """
        if not self._is_root():
            return self.root.create_group(
                posixpath.join(self.group_index, name), exist_ok=exist_ok
            )
        name = filter_name(name)
        if name in self._groups:
            if not exist_ok:
                raise TensorGroupAlreadyExistsError(name)
            return self[name]
        return self._create_group(name)

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
            scheduler (str): The scheduler to be used for rechunking. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar If ``True`` (default).
        """

        if tensors is None:
            tensors = list(self.tensors)
        elif isinstance(tensors, str):
            tensors = [tensors]

        # identity function that rechunks
        @hub.compute
        def rechunking(sample_in, samples_out):
            for tensor in tensors:
                samples_out[tensor].append(sample_in[tensor])

        rechunking().eval(
            self,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
            skip_ok=True,
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

    def extend(self, samples: Dict[str, Any], skip_ok: bool = False):
        """Appends multiple rows of samples to mutliple tensors at once. This method expects all tensors being updated to be of the same length.

        Args:
            samples (Dict[str, Any]): Dictionary with tensor names as keys and samples as values.
            skip_ok (bool): Skip tensors not in ``samples`` if set to True.

        Raises:
            KeyError: If any tensor in the dataset is not a key in ``samples`` and ``skip_ok`` is ``False``.
            TensorDoesNotExistError: If tensor in ``samples`` does not exist.
            ValueError: If all tensors being updated are not of the same length.
            NotImplementedError: If an error occurs while writing tiles.
            Exception: Error while attempting to rollback appends.
        """
        if isinstance(samples, Dataset):
            samples = samples.tensors
        if not samples:
            return
        n = len(samples[next(iter(samples.keys()))])
        for v in samples.values():
            if len(v) != n:
                sizes = {k: len(v) for (k, v) in samples.items()}
                raise ValueError(
                    f"Incoming samples are not of equal lengths. Incoming sample sizes: {sizes}"
                )
        [f() for f in list(self._update_hooks.values())]
        for i in range(n):
            self.append({k: v[i] for k, v in samples.items()})

    @invalid_view_op
    def append(
        self, sample: Dict[str, Any], skip_ok: bool = False, append_empty: bool = False
    ):
        """Append samples to mutliple tensors at once. This method expects all tensors being updated to be of the same length.

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

            >>> ds = hub.empty("../test/test_ds")
            >>> ds.create_tensor('data')
            Tensor(key='data')
            >>> ds.create_tensor('labels')
            Tensor(key='labels')
            >>> ds.append({"data": [1, 2, 3, 4], "labels":[0, 1, 2, 3]})

        """
        if isinstance(sample, Dataset):
            sample = sample.tensors
        if not isinstance(sample, dict):
            raise SampleAppendingError()

        skipped_tensors = [k for k in self.tensors if k not in sample]
        if skipped_tensors and not skip_ok and not append_empty:
            raise KeyError(
                f"Required tensors not provided: {skipped_tensors}. Pass either `skip_ok=True` to skip tensors or `append_empty=True` to append empty samples to unspecified tensors."
            )
        for k in sample:
            if k not in self._tensors():
                raise TensorDoesNotExistError(k)
        tensors_to_check_length = self.tensors if append_empty else sample
        if len(set(map(len, (self[k] for k in tensors_to_check_length)))) != 1:
            raise ValueError(
                "When appending using Dataset.append, all tensors being updated are expected to have the same length."
            )
        [f() for f in list(self._update_hooks.values())]
        tensors_appended = []
        with self:
            for k in self.tensors:
                if k in sample:
                    v = sample[k]
                else:
                    if skip_ok:
                        continue
                    else:
                        v = None
                try:
                    tensor = self[k]
                    enc = tensor.chunk_engine.chunk_id_encoder
                    num_chunks = enc.num_chunks
                    tensor.append(v)
                    tensors_appended.append(k)
                except Exception as e:
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
                    for k in tensors_appended:
                        try:
                            self[k].pop()
                        except Exception as e2:
                            raise Exception(
                                "Error while attepting to rollback appends"
                            ) from e2
                    raise e

    def _view_hash(self) -> str:
        """Generates a unique hash for a filtered dataset view."""
        return hash_inputs(
            self.path,
            *[e.value for e in self.index.values],
            self.pending_commit_id,
            getattr(self, "_query", None),
        )

    def _get_view_info(
        self,
        id: Optional[str] = None,
        message: Optional[str] = None,
        copy: bool = False,
    ):
        if self._view_invalid:
            raise DatasetViewSavingError(
                "This view cannot be saved as new changes were made at HEAD node after creation of this query view."
            )
        commit_id = self.commit_id
        if self.has_head_changes:
            if self._new_view_base_commit:
                commit_id = self._view_base_commit
            else:
                if self._view_base:
                    self._waiting_for_view_base_commit = True
                    uid = self._view_id
                    if uid not in self._update_hooks:

                        def update_hook():
                            self._view_invalid = True
                            self._waiting_for_view_base_commit = False
                            del self._view_base._update_hooks[uid]
                            del self._view_base._commit_hooks[uid]

                        def commit_hook():
                            self._waiting_for_view_base_commit = False
                            self._new_view_base_commit = self._view_base.commit_id
                            del self._view_base._update_hooks[uid]
                            del self._view_base._commit_hooks[uid]

                        self._view_base._update_hooks[uid] = update_hook
                        self._view_base._commit_hooks[uid] = commit_hook

                raise DatasetViewSavingError(
                    "HEAD node has uncommitted changes. Commit them before saving views."
                )
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
                lock.acquire(timeout=10, force=True)
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
        num_workers: Optional[int] = 0,
        scheduler: str = "threaded",
        unlink=True,
    ):
        """Writes the indices of this view to a vds."""
        vds._allow_view_updates = True
        try:
            with vds:
                if copy:
                    self._copy(
                        vds,
                        num_workers=num_workers,
                        scheduler=scheduler,
                        unlink=unlink,
                        create_vds_index_tensor=True,
                    )
                else:
                    vds.create_tensor(
                        "VDS_INDEX",
                        dtype="uint64",
                        create_shape_tensor=False,
                        create_id_tensor=False,
                        create_sample_info_tensor=False,
                    ).extend(list(self.index.values[0].indices(self.num_samples)))
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
        num_workers: int,
        scheduler: str,
    ):
        """Saves this view under ".queries" sub directory of same storage."""
        info = self._get_view_info(id, message, copy)
        hash = info["id"]
        path = f".queries/{hash}"
        vds = self._sub_ds(path, empty=True, verbose=False)
        self._write_vds(vds, info, copy, num_workers, scheduler)
        self._append_to_queries_json(info)
        return vds

    def _save_view_in_user_queries_dataset(
        self,
        id: Optional[str],
        message: Optional[str],
        copy: bool,
        num_workers: int,
        scheduler: str,
    ):
        """Saves this view under hub://username/queries
        Only applicable for views of hub datasets.
        """
        if len(self.index.values) > 1:
            raise NotImplementedError("Storing sub-sample slices is not supported yet.")

        username = jwt.decode(self.token, options={"verify_signature": False})["id"]

        if username == "public":
            raise DatasetViewSavingError(
                "Unable to save view for read only dataset. Login to save the view to your user account."
            )

        info = self._get_view_info(id, message, copy)
        base = self._view_base or self
        org_id, ds_name = base.org_id, base.ds_name
        hash = f"[{org_id}][{ds_name}]{info['id']}"
        info["id"] = hash
        queries_ds_path = f"hub://{username}/queries"

        try:
            queries_ds = hub.load(
                queries_ds_path,
                verbose=False,
            )  # create if doesn't exist
        except PathNotEmptyException:
            hub.delete(queries_ds_path, force=True)
            queries_ds = hub.empty(queries_ds_path, verbose=False)
        except DatasetHandlerError:
            queries_ds = hub.empty(queries_ds_path, verbose=False)

        queries_ds._unlock()  # we don't need locking as no data will be added to this ds.

        path = f"hub://{username}/queries/{hash}"

        vds = hub.empty(path, overwrite=True, verbose=False)

        self._write_vds(vds, info, copy, num_workers, scheduler)
        queries_ds._append_to_queries_json(info)

        return vds

    def _save_view_in_path(
        self,
        path: str,
        id: Optional[str],
        message: Optional[str],
        copy: bool,
        num_workers: int,
        scheduler: str,
        **ds_args,
    ):
        """Saves this view at a given dataset path"""
        if os.path.abspath(path) == os.path.abspath(self.path):
            raise DatasetViewSavingError("Rewriting parent dataset is not allowed.")
        try:
            vds = hub.empty(path, **ds_args)
        except Exception as e:
            raise DatasetViewSavingError from e
        info = self._get_view_info(id, message, copy)
        self._write_vds(vds, info, copy, num_workers, scheduler)
        return vds

    def save_view(
        self,
        message: Optional[str] = None,
        path: Optional[Union[str, pathlib.Path]] = None,
        id: Optional[str] = None,
        optimize: bool = False,
        num_workers: int = 0,
        scheduler: str = "threaded",
        verbose: bool = True,
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
                - If the user doesn't have write access to the source dataset and the source dataset is a hub cloud dataset, then the VDS is saved is saved under the user's hub account and can be accessed using ``hub.load(f"hub://{username}/queries/{query_hash}")``.
            id (Optional, str): Unique id for this view. Random id will be generated if not specified.
            optimize (bool):
                - If ``True``, the dataset view will be optimized by copying and rechunking the required data. This is necessary to achieve fast streaming speeds when training models using the dataset view. The optimization process will take some time, depending on the size of the data.
                - You can also choose to optimize the saved view later by calling its :meth:`ViewEntry.optimize` method.
            num_workers (int): Number of workers to be used for optimization process. Applicable only if ``optimize=True``. Defaults to 0.
            scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded', 'processed' and 'ray'. Only applicable if ``optimize=True``. Defaults to 'threaded'.
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.
            ds_args (dict): Additional args for creating VDS when path is specified. (See documentation for :func:`hub.dataset()`)

        Returns:
            str: Path to the saved VDS.

        Raises:
            ReadOnlyModeError: When attempting to save a view inplace and the user doesn't have write access.
            DatasetViewSavingError: If HEAD node has uncommitted changes.

        Note:
            Specifying ``path`` makes the view external. External views cannot be accessed using the parent dataset's :func:`Dataset.get_view`,
            :func:`Dataset.load_view`, :func:`Dataset.delete_view` methods. They have to be loaded using :func:`hub.load`.
        """
        return self._save_view(
            path,
            id,
            message,
            optimize,
            num_workers,
            scheduler,
            verbose,
            False,
            **ds_args,
        )

    def _save_view(
        self,
        path: Optional[Union[str, pathlib.Path]] = None,
        id: Optional[str] = None,
        message: Optional[str] = None,
        optimize: bool = False,
        num_workers: int = 0,
        scheduler: str = "threaded",
        verbose: bool = True,
        _ret_ds: bool = False,
        **ds_args,
    ) -> Union[str, Any]:
        """Saves a dataset view as a virtual dataset (VDS)

        Args:
            path (Optional, str, pathlib.Path): If specified, the VDS will saved as a standalone dataset at the specified path. If not,
                the VDS is saved under `.queries` subdirectory of the source dataset's storage. If the user doesn't have
                write access to the source dataset and the source dataset is a hub cloud dataset, then the VDS is saved
                is saved under the user's hub account and can be accessed using hub.load(f"hub://{username}/queries/{query_hash}").
            id (Optional, str): Unique id for this view.
            message (Optional, message): Custom user message.
            optimize (bool): Whether the view should be optimized by copying the required data. Default False.
            num_workers (int): Number of workers to be used if `optimize` is True.
            scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Only applicable if ``optimize=True``. Defaults to 'threaded'.
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.
            _ret_ds (bool): If ``True``, the VDS is retured as such without converting it to a view. If ``False``, the VDS path is returned.
                Default False.
            ds_args (dict): Additional args for creating VDS when path is specified. (See documentation for `hub.dataset()`)

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
        if vds is None:
            if path is None:
                if isinstance(self, MemoryProvider):
                    raise NotImplementedError(
                        "Saving views inplace is not supported for in-memory datasets."
                    )
                if self.read_only and not (self._view_base or self)._locked_out:
                    if isinstance(self, hub.core.dataset.HubCloudDataset):
                        vds = self._save_view_in_user_queries_dataset(
                            id, message, optimize, num_workers, scheduler
                        )
                    else:
                        raise ReadOnlyModeError(
                            "Cannot save view in read only dataset. Speicify a path to save the view in a different location."
                        )
                else:
                    vds = self._save_view_in_subdir(
                        id, message, optimize, num_workers, scheduler
                    )
            else:
                vds = self._save_view_in_path(
                    path, id, message, optimize, num_workers, scheduler, **ds_args
                )
        if verbose:
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
            self._parent_dataset
            if (inherit_creds and self._parent_dataset)
            else hub.load(
                self.info["source-dataset"], verbose=False, creds=creds, read_only=True
            )
        )
        try:
            orig_index = ds.index
            ds.index = Index()
            ds._checkout(commit_id, verbose=False)
            first_index_subscriptable = self.info.get("first-index-subscriptable", True)
            if first_index_subscriptable:
                index_entries = [
                    IndexEntry(self.VDS_INDEX.numpy().reshape(-1).tolist())
                ]
            else:
                index_entries = [IndexEntry(int(self.VDS_INDEX.numpy()))]
            sub_sample_index = self.info.get("sub-sample-index")
            if sub_sample_index:
                index_entries += Index.from_json(sub_sample_index).values
            ret = ds[Index(index_entries)]
            ret._vds = self
            return ret
        finally:
            ds.index = orig_index

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

    @staticmethod
    def _get_queries_ds_from_user_account():
        username = get_user_name()
        if username == "public":
            return
        try:
            return hub.load(f"hub://{username}/queries", verbose=False)
        except DatasetHandlerError:
            return

    def _read_queries_json_from_user_account(self):
        queries_ds = Dataset._get_queries_ds_from_user_account()
        if not queries_ds:
            return [], None
        return (
            list(
                filter(
                    lambda x: x["source-dataset"] == self.path,
                    queries_ds._read_queries_json(),
                )
            ),
            queries_ds,
        )

    def get_views(self, commit_id: Optional[str] = None) -> List[ViewEntry]:
        """Returns list of views stored in this Dataset.

        Args:
            commit_id (str, optional): - Commit from which views should be returned.
                - If not specified, views from current commit is returned.
                - If not specified, views from the currently checked out commit will be returned.

        Returns:
            List[ViewEntry]: List of :class:`ViewEntry` instances.
        """
        commit_id = commit_id or self.commit_id
        queries = self._read_queries_json()
        f = lambda x: x["source-dataset-version"] == commit_id
        ret = map(
            partial(ViewEntry, dataset=self),
            filter(f, queries),
        )

        if self.path.startswith("hub://"):
            queries, qds = self._read_queries_json_from_user_account()
            if queries:
                ret = chain(
                    ret,
                    map(
                        partial(
                            ViewEntry, dataset=qds, source_dataset=self, external=True
                        ),
                        filter(f, queries),
                    ),
                )
        return list(ret)

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
        if self.path.startswith("hub://"):
            queries, qds = self._read_queries_json_from_user_account()
            for q in queries:
                if q["id"] == f"[{self.org_id}][{self.ds_name}]{id}":
                    return ViewEntry(q, qds, self, True)
        raise KeyError(f"No view with id {id} found in the dataset.")

    def load_view(
        self,
        id: str,
        optimize: Optional[bool] = False,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: Optional[bool] = True,
    ):
        """Loads the view and returns the :class:`~hub.core.dataset.dataset.Dataset` by id. Equivalent to ds.get_view(id).load().

        Args:
            id (str): id of the view to be loaded.
            optimize (bool): If ``True``, the dataset view is optimized by copying and rechunking the required data before loading. This is
                necessary to achieve fast streaming speeds when training models using the dataset view. The optimization process will
                take some time, depending on the size of the data.
            num_workers (int): Number of workers to be used for the optimization process. Only applicable if `optimize=True`. Defaults to 0.
            scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Only applicable if `optimize=True`. Defaults to 'threaded'.
            progressbar (bool): Whether to use progressbar for optimization. Only applicable if `optimize=True`. Defaults to True.

        Returns:
            Dataset: The loaded view.

        Raises:
            KeyError: if view with given id does not exist.
        """
        if optimize:
            return (
                self.get_view(id)
                .optimize(
                    num_workers=num_workers,
                    scheduler=scheduler,
                    progressbar=progressbar,
                )
                .load()
            )
        return self.get_view(id).load()

    def delete_view(self, id: str):
        """Deletes the view with given view id.

        Args:
            id (str): Id of the view to delete.

        Raises:
            KeyError: if view with given id does not exist.
        """
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
        except Exception:
            pass
        if self.path.startswith("hub://"):
            qds = Dataset._get_queries_ds_from_user_account()
            if qds:
                with qds._lock_queries_json():
                    qjson = qds._read_queries_json()
                    for i, q in enumerate(qjson):
                        if (
                            q["source-dataset"] == self.path
                            and q["id"] == f"[{self.org_id}][{self.ds_name}]{id}"
                        ):
                            qjson.pop(i)
                            qds.base_storage.subdir(
                                ".queries/" + (q.get("path") or q["id"])
                            ).clear()
                            qds._write_queries_json(qjson)
                            return
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
    ):
        """Loads a nested dataset. Internal.

        Args:
            path (str): Path to sub directory
            empty (bool): If ``True``, all contents of the sub directory is cleared before initializing the sub dataset.
            memory_cache_size (int): Memory cache size for the sub dataset.
            local_cache_size (int): Local storage cache size for the sub dataset.
            read_only (bool): Loads the sub dataset in read only mode if ``True``. Default ``False``.
            lock (bool): Whether the dataset should be locked for writing. Only applicable for s3, hub and gcs datasets. No effect if ``read_only=True``.
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.

        Returns:
            Sub dataset

        Note:
            Virtual datasets are returned as such, they are not converted to views.
        """
        sub_storage = self.base_storage.subdir(path)

        if empty:
            sub_storage.clear()

        if self.path.startswith("hub://"):
            path = posixpath.join(self.path, path)
            cls = hub.core.dataset.HubCloudDataset
        else:
            path = sub_storage.root
            cls = hub.core.dataset.Dataset

        ret = cls(
            generate_chain(
                sub_storage,
                memory_cache_size * MB,
                local_cache_size * MB,
            ),
            path=path,
            token=self._token,
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
        append_f: str,
        update_f: Optional[str] = None,
        flatten_sequence: Optional[bool] = None,
    ):
        """Internal. Links a source tensor to a destination tensor. Appends / updates made to the source tensor will be reflected in the destination tensor.

        Args:
            src (str): Name of the source tensor.
            dest (str): Name of the destination tensor.
            append_f (str): Name of the linked tensor transform to be used for appending items to the destination tensor. This transform should be defined in `hub.core.tensor_link` module.
            update_f (str): Name of the linked tensor transform to be used for updating items in the destination tensor. This transform should be defined in `hub.core.tensor_link` module.
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
        src_tensor.meta.add_link(dest_key, append_f, update_f, flatten_sequence)
        self.storage.maybe_flush()

    def _resolve_tensor_list(self, keys: List[str]) -> List[str]:
        ret = []
        for k in keys:
            fullpath = posixpath.join(self.group_index, k)
            if (
                self.version_state["tensor_names"].get(fullpath)
                in self.version_state["full_tensors"]
            ):
                ret.append(k)
            else:
                if fullpath[-1] != "/":
                    fullpath = fullpath + "/"
                hidden = self.meta.hidden_tensors
                ret += filter(
                    lambda t: t.startswith(fullpath) and t not in hidden,
                    self.version_state["tensor_names"],
                )
        return ret

    def _copy(
        self,
        dest: Union[str, pathlib.Path],
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
    ):
        """Copies this dataset or dataset view to `dest`. Version control history is not included.

        Args:
            dest (str, pathlib.Path): Destination dataset or path to copy to. If a Dataset instance is provided, it is expected to be empty.
            tensors (List[str], optional): Names of tensors (and groups) to be copied. If not specified all tensors are copied.
            overwrite (bool): If ``True`` and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            creds (dict, Optional): creds required to create / overwrite datasets at `dest`.
            token (str, Optional): token used to for fetching credentials to `dest`.
            num_workers (int): The number of workers to use for copying. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used for copying. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar If ``True`` (default).
            public (bool): Defines if the dataset will have public access. Applicable only if Hub cloud storage is used and a new Dataset is being created. Defaults to False.
            unlink (bool): Whether to copy the data from source for linked tensors. Does not apply for linked video tensors.
            create_vds_index_tensor (bool): If ``True``, a hidden tensor called "VDS_INDEX" is created which contains the sample indices in the source view.

        Returns:
            Dataset: New dataset object.

        Raises:
            DatasetHandlerError: If a dataset already exists at destination path and overwrite is False.
        """
        if isinstance(dest, str):
            path = dest
        else:
            path = dest.path

        report_params = {
            "Tensors": tensors,
            "Overwrite": overwrite,
            "Num_Workers": num_workers,
            "Scheduler": scheduler,
            "Progressbar": progressbar,
            "Public": public,
        }

        if path.startswith("hub://"):
            report_params["Dest"] = path
        feature_report_path(self.path, "copy", report_params, token=token)

        dest_ds = hub.api.dataset.dataset._like(
            dest,
            self,
            tensors=tensors,
            creds=creds,
            token=token,
            overwrite=overwrite,
            public=public,
            unlink=[
                t
                for t in self.tensors
                if (
                    self.tensors[t].base_htype != "video"
                    or hub.constants._UNLINK_VIDEOS
                )
            ]
            if unlink
            else False,
        )

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
            for tensor in dest_ds.tensors:
                src = self[tensor]
                copy_f = (
                    (
                        _copy_tensor_unlinked_partial_sample
                        if len(self.index) > 1
                        else _copy_tensor_unlinked_full_sample
                    )
                    if unlink
                    and src.is_link
                    and (src.base_htype != "video" or hub.constants._UNLINK_VIDEOS)
                    else _copy_tensor
                )
                if progressbar:
                    sys.stderr.write(f"Copying tensor: {tensor}.\n")
                hub.compute(copy_f, name="tensor copy transform")(
                    tensor_name=tensor
                ).eval(
                    self,
                    dest_ds,
                    num_workers=num_workers,
                    scheduler=scheduler,
                    progressbar=progressbar,
                    skip_ok=True,
                    check_lengths=False,
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
        return dest_ds

    def copy(
        self,
        dest: Union[str, pathlib.Path],
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        creds=None,
        token=None,
        num_workers: int = 0,
        scheduler="threaded",
        progressbar=True,
        public: bool = False,
    ):
        """Copies this dataset or dataset view to ``dest``. Version control history is not included.

        Args:
            dest (str, pathlib.Path): Destination dataset or path to copy to. If a Dataset instance is provided, it is expected to be empty.
            tensors (List[str], optional): Names of tensors (and groups) to be copied. If not specified all tensors are copied.
            overwrite (bool): If ``True`` and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            creds (dict, Optional): creds required to create / overwrite datasets at `dest`.
            token (str, Optional): token used to for fetching credentials to `dest`.
            num_workers (int): The number of workers to use for copying. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used for copying. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar If ``True`` (default).
            public (bool): Defines if the dataset will have public access. Applicable only if Hub cloud storage is used and a new Dataset is being created. Defaults to False.

        Returns:
            Dataset: New dataset object.

        Raises:
            DatasetHandlerError: If a dataset already exists at destination path and overwrite is False.
        """
        return self._copy(
            dest,
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
    def reset(self):
        """Resets the uncommitted changes present in the branch.

        Note:
            The uncommitted data is deleted from underlying storage, this is not a reversible operation.
        """
        storage, version_state = self.storage, self.version_state
        if version_state["commit_node"].children:
            print("You are not at the head node of the branch, cannot reset.")
            return
        if not self.has_head_changes:
            print("There are no uncommitted changes on this branch.")
            return

        # delete metas first
        self._delete_metas()

        if self.commit_id is None:
            storage.clear()
            self._populate_meta()
        else:
            prefix = "/".join(("versions", self.pending_commit_id))
            storage.clear(prefix=prefix)
            src_id, dest_id = self.commit_id, self.pending_commit_id

            # by doing this checkout, we get list of tensors in previous commit, which is what we require for copying metas and create_commit_chunk_set
            self.checkout(src_id)
            copy_metas(src_id, dest_id, storage, version_state)
            create_commit_chunk_sets(dest_id, storage, version_state)
            self.checkout(dest_id)
        load_meta(self)
        self._info = None
        self._ds_diff = None

    def _delete_metas(self):
        """Deletes all metas in the dataset."""
        commit_id = self.pending_commit_id
        meta_keys = [get_dataset_meta_key(commit_id)]
        meta_keys.append(get_dataset_diff_key(commit_id))
        meta_keys.append(get_dataset_info_key(commit_id))

        for tensor in self.tensors:
            meta_keys.append(get_tensor_meta_key(commit_id, tensor))
            meta_keys.append(get_tensor_tile_encoder_key(commit_id, tensor))
            meta_keys.append(get_tensor_info_key(commit_id, tensor))
            meta_keys.append(get_tensor_commit_chunk_set_key(commit_id, tensor))
            meta_keys.append(get_tensor_commit_diff_key(commit_id, tensor))
            meta_keys.append(get_chunk_id_encoder_key(commit_id, tensor))
            meta_keys.append(get_sequence_encoder_key(commit_id, tensor))

        for key in meta_keys:
            try:
                del self.storage[key]
            except KeyError:
                pass

    def add_creds_key(self, creds_key: str, managed: bool = False):
        """Adds a new creds key to the dataset. These keys are used for tensors that are linked to external data.

        Examples:

            >>> # create/load a dataset
            >>> ds = hub.empty("path/to/dataset")
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

    def populate_creds(self, creds_key: str, creds: dict):
        """Populates the creds key added in add_creds_key with the given creds. These creds are used to fetch the external data.
        This needs to be done everytime the dataset is reloaded for datasets that contain links to external data.

        Examples:

            >>> # create/load a dataset
            >>> ds = hub.dataset("path/to/dataset")
            >>> # add a new creds key
            >>> ds.add_creds_key("my_s3_key")
            >>> # populate the creds
            >>> ds.populate_creds("my_s3_key", {"aws_access_key_id": "my_access_key", "aws_secret_access_key": "my_secret_key"})

        """
        self.link_creds.populate_creds(creds_key, creds)

    def update_creds_key(self, old_creds_key: str, new_creds_key: str):
        """Replaces the old creds key with the new creds key. This is used to replace the creds key used for external data."""
        replaced_index = self.link_creds.replace_creds(old_creds_key, new_creds_key)
        save_link_creds(self.link_creds, self.storage, replaced_index=replaced_index)

    def change_creds_management(self, creds_key: str, managed: bool):
        """Changes the management status of the creds key.

        Args:
            creds_key (str): The key whose management status is to be changed.
            managed (bool): The target management status. If ``True``, the creds corresponding to the key will be fetched from activeloop platform.

        Raises:
            ValueError: If the dataset is not connected to activeloop platform.
            KeyError: If the creds key is not present in the dataset.

        Examples:

            >>> # create/load a dataset
            >>> ds = hub.dataset("path/to/dataset")
            >>> # add a new creds key
            >>> ds.add_creds_key("my_s3_key")
            >>> # Populate the name added with creds dictionary
            >>> # These creds are only present temporarily and will have to be repopulated on every reload
            >>> ds.populate_creds("my_s3_key", {})
            >>> # Change the management status of the key to True. Before doing this, ensure that the creds have been created on activeloop platform
            >>> # Now, this key will no longer use the credentials populated in the previous step but will instead fetch them from activeloop platform
            >>> # These creds don't have to be populated again on every reload and will be fetched every time the dataset is loaded
            >>> ds.change_creds_management("my_s3_key", True)

        """
        raise ValueError(
            "Managed creds are not supported for datasets that are not connected to activeloop platform."
        )

    def get_creds_keys(self) -> List[str]:
        """Returns the list of creds keys added to the dataset. These are used to fetch external data in linked tensors"""
        return self.link_creds.creds_keys

    def visualize(
        self, width: Union[int, str, None] = None, height: Union[int, str, None] = None
    ):
        """
        Visualizes the dataset in the Jupyter notebook.

        Args:
            width: Union[int, str, None] Optional width of the visualizer canvas.
            height: Union[int, str, None] Optional height of the visualizer canvas.

        Raises:
            Exception: If the dataset is not a hub cloud dataset and the visualization is attempted in colab.
        """
        from hub.visualizer import visualize

        hub_reporter.feature_report(feature_name="visualize", parameters={})
        if is_colab():
            raise Exception("Cannot visualize non hub cloud dataset in Colab.")
        else:
            visualize(self.storage, width=width, height=height)

    def __contains__(self, tensor: str):
        return tensor in self.tensors

    def _optimize_saved_view(
        self,
        id: str,
        external=False,
        unlink=True,
        num_workers=0,
        scheduler="threaded",
        progressbar=True,
    ):
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
            vds = self._sub_ds(".queries/" + path, verbose=False)
            view = vds._get_view(not external)
            new_path = path + "_OPTIMIZED"
            optimized = self._sub_ds(".queries/" + new_path, empty=True, verbose=False)
            view._copy(
                optimized,
                overwrite=True,
                unlink=unlink,
                create_vds_index_tensor=True,
                num_workers=num_workers,
                scheduler=scheduler,
                progressbar=progressbar,
            )
            optimized.info.update(vds.info.__getstate__())
            optimized.info["virtual-datasource"] = False
            optimized.info["path"] = new_path
            optimized.flush()
            info["virtual-datasource"] = False
            info["path"] = new_path
            self._write_queries_json(qjson)
        vds.base_storage.disable_readonly()
        try:
            vds.base_storage.clear()
        except Exception as e:
            warnings.warn(
                f"Error while deleting old view after writing optimized version: {e}"
            )
        return info

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

    @invalid_view_op
    def pop(self, index: Optional[int] = None):
        """
        Removes a sample from all the tensors of the dataset.
        For any tensor if the index >= len(tensor), the sample won't be popped from it.

        Args:
            index (int, Optional): The index of the sample to be removed. If it is ``None``, the index becomes the ``length of the longest tensor - 1``.

        Raises:
            IndexError: If the index is out of range.
        """
        max_len = max((t.num_samples for t in self.tensors.values()), default=0)
        if max_len == 0:
            raise IndexError("Can't pop from empty dataset.")

        if index is None:
            index = max_len - 1

        if index < 0:
            raise IndexError("Pop doesn't support negative indices.")
        elif index >= max_len:
            raise IndexError(
                f"Index {index} is out of range. The longest tensor has {max_len} samples."
            )

        for tensor in self.tensors.values():
            if tensor.num_samples > index:
                tensor.pop(index)

    @property
    def is_view(self) -> bool:
        """Returns ``True`` if this dataset is a view and ``False`` otherwise."""
        return (
            not self.index.is_trivial()
            or hasattr(self, "_vds")
            or hasattr(self, "_view_entry")
        )


def _copy_tensor(sample_in, sample_out, tensor_name):
    sample_out[tensor_name].append(sample_in[tensor_name])


def _copy_tensor_unlinked_full_sample(sample_in, sample_out, tensor_name):
    sample_out[tensor_name].append(
        sample_in[tensor_name].chunk_engine.get_hub_read_sample(
            sample_in.index.values[0].value
        )
    )


def _copy_tensor_unlinked_partial_sample(sample_in, sample_out, tensor_name):
    sample_out[tensor_name].append(sample_in[tensor_name].numpy())
