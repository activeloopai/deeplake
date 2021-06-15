"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import warnings
from hub_v1.api.versioning import VersionNode
import os
import posixpath
import collections.abc as abc
import json
import sys
from typing import Iterable
import traceback
from collections import defaultdict
import numpy as np
from PIL import Image as im, ImageChops

import fsspec
from fsspec.spec import AbstractFileSystem
import numcodecs
import numcodecs.lz4
import numcodecs.zstd

from hub_v1.schema.features import (
    Primitive,
    Tensor,
    SchemaDict,
    featurify,
)
from hub_v1.log import logger
import hub_v1.store.pickle_s3_storage

from hub_v1.api.datasetview import DatasetView
from hub_v1.api.objectview import ObjectView
from hub_v1.api.tensorview import TensorView
from hub_v1.api.dataset_utils import (
    create_numpy_dict,
    generate_hash,
    get_value,
    slice_split,
    str_to_int,
    _copy_helper,
    _get_compressor,
    _get_dynamic_tensor_dtype,
    _store_helper,
    check_class_label,
    same_schema,
)

import hub_v1.schema.serialize
import hub_v1.schema.deserialize
from hub_v1.schema.features import flatten
from hub_v1 import auto

from hub_v1.store.dynamic_tensor import DynamicTensor
from hub_v1.store.store import get_fs_and_path, get_storage_map
from hub_v1.exceptions import (
    AddressNotFound,
    HubDatasetNotFoundException,
    NotHubDatasetToOverwriteException,
    NotHubDatasetToAppendException,
    OutOfBoundsError,
    ReadModeException,
    ShapeArgumentNotFoundException,
    SchemaArgumentNotFoundException,
    ModuleNotInstalledException,
    ShapeLengthException,
    VersioningNotSupportedException,
    WrongUsernameException,
    InvalidVersionInfoException,
    SchemaMismatchException,
)
from hub_v1.store.metastore import MetaStorage
from hub_v1.client.hub_control import HubControlClient
from hub_v1.schema import Audio, BBox, ClassLabel, Image, Sequence, Text, Video
from hub_v1.utils import norm_cache, norm_shape, _tuple_product
from hub_v1 import defaults
import pickle

import sys

sys.modules["hub"] = hub_v1


def get_file_count(fs: fsspec.AbstractFileSystem, path):
    return len(fs.listdir(path, detail=False))


class Dataset:
    def __init__(
        self,
        url: str,
        mode: str = None,
        shape=None,
        schema=None,
        token=None,
        fs=None,
        fs_map=None,
        meta_information=dict(),
        cache: int = defaults.DEFAULT_MEMORY_CACHE_SIZE,
        storage_cache: int = defaults.DEFAULT_STORAGE_CACHE_SIZE,
        lock_cache=True,
        tokenizer=None,
        lazy: bool = True,
        public: bool = True,
        name: str = None,
    ):
        """| Open a new or existing dataset for read/write

        Parameters
        ----------
        url: str
            The url where dataset is located/should be created
        mode: str, optional (default to "a")
            Python way to tell whether dataset is for read or write (ex. "r", "w", "a")
        shape: tuple, optional
            Tuple with (num_samples,) format, where num_samples is number of samples
        schema: optional
            Describes the data of a single sample. Hub schemas are used for that
            Required for 'a' and 'w' modes
        token: str or dict, optional
            If url is refering to a place where authorization is required,
            token is the parameter to pass the credentials, it can be filepath or dict
        fs: optional
        fs_map: optional
        meta_information: optional ,give information about dataset in a dictionary.
        cache: int, optional
            Size of the memory cache. Default is 64MB (2**26)
            if 0, False or None, then cache is not used
        storage_cache: int, optional
            Size of the storage cache. Default is 256MB (2**28)
            if 0, False or None, then storage cache is not used
        lock_cache: bool, optional
            Lock the cache for avoiding multiprocessing errors
        lazy: bool, optional
            Setting this to False will stop lazy computation and will allow items to be accessed without .compute()
        public: bool, optional
            only applicable if using hub storage, ignored otherwise
            setting this to False allows only the user who created it to access the dataset and
            the dataset won't be visible in the visualizer to the public
        name: str, optional
            only applicable when using hub storage, this is the name that shows up on the visualizer
        """

        shape = norm_shape(shape)
        if len(shape) != 1:
            raise ShapeLengthException()

        storage_cache = norm_cache(storage_cache) if cache else 0
        cache = norm_cache(cache)

        schema: SchemaDict = featurify(schema) if schema else None

        self._url = url
        self._token = token
        self.tokenizer = tokenizer
        self.lazy = lazy
        self._name = name

        self._fs, self._path = (
            (fs, url) if fs else get_fs_and_path(self._url, token=token, public=public)
        )
        self._cache = cache
        self._storage_cache = storage_cache
        self.lock_cache = lock_cache
        self.verison = "1.x"
        mode = self._get_mode(mode, self._fs)
        self._mode = mode
        needcreate = self._check_and_prepare_dir()
        fs_map = fs_map or get_storage_map(
            self._fs, self._path, cache, lock=lock_cache, storage_cache=storage_cache
        )
        self._fs_map = fs_map
        self._meta_information = meta_information
        self.username = None
        self.dataset_name = None
        if not needcreate:
            self.meta = json.loads(fs_map[defaults.META_FILE].decode("utf-8"))
            self._name = self.meta.get("name") or None
            self._shape = tuple(self.meta["shape"])
            self._schema = hub_v1.schema.deserialize.deserialize(self.meta["schema"])
            self._meta_information = self.meta.get("meta_info") or dict()
            self._flat_tensors = tuple(flatten(self._schema))
            try:
                version_info = pickle.loads(fs_map[defaults.VERSION_INFO])
                self._branch_node_map = version_info.get("branch_node_map")
                self._commit_node_map = version_info.get("commit_node_map")
                self._chunk_commit_map = version_info.get("chunk_commit_map")
                if not (
                    self._branch_node_map
                    and self._commit_node_map
                    and self._chunk_commit_map
                ):
                    raise InvalidVersionInfoException()
                self._branch = "master"
                self._version_node = self._branch_node_map[self._branch]
                self._commit_id = self._version_node.commit_id
            except KeyError:
                self._commit_id = None
                self._branch = None
                self._version_node = None
                self._branch_node_map = None
                self._commit_node_map = None
                self._chunk_commit_map = None
            except InvalidVersionInfoException:
                self._commit_id = None
                self._branch = None
                self._version_node = None
                self._branch_node_map = None
                self._commit_node_map = None
                self._chunk_commit_map = None

            self._tensors = dict(self._open_storage_tensors())

            if shape != (None,) and shape != self._shape:
                raise TypeError(
                    f"Shape stored previously [{self._shape}]  and shape in arguments [{shape}] are !=, use mode='w' to overwrite dataset"
                )
            if schema is not None and not same_schema(schema, self._schema):
                raise SchemaMismatchException()

        else:
            if shape[0] is None:
                raise ShapeArgumentNotFoundException()
            if schema is None:
                raise SchemaArgumentNotFoundException()
            try:
                if shape is None:
                    raise ShapeArgumentNotFoundException()
                if schema is None:
                    raise SchemaArgumentNotFoundException()
                self._schema = schema
                self._shape = tuple(shape)
                self.meta = self._store_meta()
                self._meta_information = meta_information
                self._flat_tensors = tuple(flatten(self.schema))

                self._commit_id = generate_hash()
                self._branch = "master"
                self._version_node = VersionNode(self._commit_id, self._branch)
                self._branch_node_map = {self._branch: self._version_node}
                self._commit_node_map = {self._commit_id: self._version_node}
                self._chunk_commit_map = {
                    path: defaultdict(set) for schema, path in self._flat_tensors
                }
                self._tensors = dict(self._generate_storage_tensors())
            except Exception as e:
                try:
                    self.close()
                except Exception:
                    pass
                self._fs.rm(self._path, recursive=True)
                logger.error("Deleting the dataset " + traceback.format_exc() + str(e))
                raise
        self.flush()
        self.indexes = list(range(self._shape[0]))

        if self._path.startswith("s3://snark-hub-dev/") or self._path.startswith(
            "s3://snark-hub/"
        ):
            subpath = self._path[5:]
            spl = subpath.split("/")
            if len(spl) < 4:
                raise ValueError("Invalid Path for dataset")
            self.username = spl[-2]
            self.dataset_name = spl[-1]
            if needcreate:
                HubControlClient().create_dataset_entry(
                    self.username, self.dataset_name, self.meta, public=public
                )

    @property
    def mode(self):
        return self._mode

    @property
    def url(self):
        return self._url

    @property
    def shape(self):
        return self._shape

    @property
    def name(self):
        return self._name

    @property
    def token(self):
        return self._token

    @property
    def cache(self):
        return self._cache

    @property
    def storage_cache(self):
        return self._storage_cache

    @property
    def schema(self):
        return self._schema

    @property
    def meta_information(self):
        return self._meta_information

    def _store_meta(self) -> dict:
        meta = {
            "shape": self._shape,
            "schema": hub_v1.schema.serialize.serialize(self._schema),
            "version": 1,
            "meta_info": self._meta_information or dict(),
            "name": self._name,
        }

        self._fs_map[defaults.META_FILE] = bytes(json.dumps(meta), "utf-8")
        return meta

    def _store_version_info(self) -> dict:
        if self._commit_id is not None:
            d = {
                "branch_node_map": self._branch_node_map,
                "commit_node_map": self._commit_node_map,
                "chunk_commit_map": self._chunk_commit_map,
            }
            self._fs_map[defaults.VERSION_INFO] = pickle.dumps(d)

    def commit(self, message: str = "") -> str:
        """| Saves the current state of the dataset and returns the commit id.
        Checks out automatically to an auto branch if the current commit is not the head of the branch

        Only saves the dataset without any version control information if the dataset was created before Hub v1.3.0

        Parameters
        ----------
        message: str, optional
            The commit message to store along with the commit
        """
        if self._commit_id is None:
            warnings.warn(
                "This dataset was created before version control, it does not support it. commit will behave same as flush"
            )
            self.flush()
        elif "r" in self._mode:
            raise ReadModeException("commit")
        else:
            self._auto_checkout()
            stored_commit_id = self._commit_id
            self._commit_id = generate_hash()
            new_node = VersionNode(self._commit_id, self._branch)
            self._version_node.insert(new_node, message)
            self._version_node = new_node
            self._branch_node_map[self._branch] = new_node
            self._commit_node_map[self._commit_id] = new_node
            self.flush()
            return stored_commit_id

    def checkout(self, address: str, create: bool = False) -> str:
        """| Changes the state of the dataset to the address mentioned. Creates a new branch if address isn't a commit id or branch name and create is True.
        Always checks out to the head of a branch if the address specified is a branch name.

        Returns the commit id of the commit that has been switched to.

        Only works if dataset was created on or after Hub v1.3.0

        Parameters
        ----------
        address: str
            The branch name or commit id to checkout to
        create: bool, optional
            Specifying create as True creates a new branch from the current commit if the address isn't an existing branch name or commit id
        """
        if self._commit_id is None:
            raise VersioningNotSupportedException("checkout")
        self.flush()
        if address in self._branch_node_map.keys():
            self._branch = address
            self._version_node = self._branch_node_map[address]
            self._commit_id = self._version_node.commit_id
        elif address in self._commit_node_map.keys():
            self._version_node = self._commit_node_map[address]
            self._branch = self._version_node.branch
            self._commit_id = self._version_node.commit_id
        elif create:
            if "r" in self._mode:
                raise ReadModeException("checkout to create new branch")
            self._branch = address
            new_commit_id = generate_hash()
            new_node = VersionNode(new_commit_id, self._branch)
            if not self._version_node.children:
                for key in self.keys:
                    self._tensors[key].fs_map.copy_all_chunks(
                        self._commit_id, new_commit_id
                    )
                if self._version_node.parent is not None:
                    self._version_node.parent.insert(
                        new_node, f"switched to new branch {address}"
                    )
            else:
                self._version_node.insert(new_node, f"switched to new branch {address}")
            self._version_node = new_node
            self._commit_id = new_commit_id
            self._branch_node_map[self._branch] = new_node
            self._commit_node_map[self._commit_id] = new_node
            self.flush()
        else:
            raise AddressNotFound(address)
        return self._commit_id

    def _auto_checkout(self):
        """| Automatically checks out to a new branch if the current commit is not at the head of a branch"""
        if self._version_node and self._version_node.children:
            branch_name = f"'auto-{generate_hash()}'"
            print(
                f"automatically checking out to new branch {branch_name} as not at the head of branch {self._branch}"
            )
            self.checkout(branch_name, True)

    def log(self):
        """| Prints the commits in the commit tree before the current commit
        Only works if dataset was created on or after Hub v1.3.0
        """
        if self._commit_id is None:
            raise VersioningNotSupportedException("log")
        current_node = (
            self._version_node.parent
            if not self._version_node.children
            else self._version_node
        )
        print(f"\n Current Branch: {self._branch}\n")
        while current_node:
            print(f"{current_node}\n")
            current_node = current_node.parent

    def _check_and_prepare_dir(self):
        """
        Checks if input data is ok.
        Creates or overwrites dataset folder.
        Returns True dataset needs to be created opposed to read.
        """
        fs, path, mode = self._fs, self._path, self._mode
        if path.startswith("s3://"):
            with open(os.path.expanduser("~/.activeloop/store"), "rb") as f:
                stored_username = json.load(f)["_id"]
            current_username = path.split("/")[-2]
            if stored_username != current_username:
                try:
                    fs.listdir(path)
                except BaseException:
                    raise WrongUsernameException(stored_username)
        meta_path = posixpath.join(path, defaults.META_FILE)
        try:
            # Update boto3 cache
            fs.ls(path, detail=False, refresh=True)
        except Exception:
            pass
        exist_meta = fs.exists(meta_path)
        if exist_meta:
            if "w" in mode:
                fs.rm(path, recursive=True)
                fs.makedirs(path)
                return True
            return False
        else:
            if "r" in mode:
                raise HubDatasetNotFoundException(path)
            exist_dir = fs.exists(path)
            if not exist_dir:
                fs.makedirs(path)
            elif get_file_count(fs, path) > 0:
                if "w" in mode:
                    raise NotHubDatasetToOverwriteException()
                else:
                    raise NotHubDatasetToAppendException()
            return True

    def _generate_storage_tensors(self):
        for t in self._flat_tensors:
            t_dtype, t_path = t
            path = posixpath.join(self._path, t_path[1:])
            self._fs.makedirs(posixpath.join(path, "--dynamic--"))
            yield t_path, DynamicTensor(
                fs_map=MetaStorage(
                    t_path,
                    get_storage_map(
                        self._fs,
                        path,
                        self._cache,
                        self.lock_cache,
                        storage_cache=self._storage_cache,
                    ),
                    self._fs_map,
                    self,
                ),
                mode=self._mode,
                shape=self._shape + t_dtype.shape,
                max_shape=self._shape + t_dtype.max_shape,
                dtype=_get_dynamic_tensor_dtype(t_dtype),
                chunks=t_dtype.chunks,
                compressor=_get_compressor(t_dtype.compressor),
            )

    def _open_storage_tensors(self):
        for t in self._flat_tensors:
            t_dtype, t_path = t
            path = posixpath.join(self._path, t_path[1:])
            yield t_path, DynamicTensor(
                fs_map=MetaStorage(
                    t_path,
                    get_storage_map(
                        self._fs,
                        path,
                        self._cache,
                        self.lock_cache,
                        storage_cache=self._storage_cache,
                    ),
                    self._fs_map,
                    self,
                ),
                mode=self._mode,
                # FIXME We don't need argument below here
                shape=self._shape + t_dtype.shape,
            )

    def __getitem__(self, slice_):
        """| Gets a slice or slices from dataset
        | Usage:
        >>> return ds["image", 5, 0:1920, 0:1080, 0:3].compute() # returns numpy array
        >>> images = ds["image"]
        >>> return images[5].compute() # returns numpy array
        >>> images = ds["image"]
        >>> image = images[5]
        >>> return image[0:1920, 0:1080, 0:3].compute()
        """
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        if not subpath:
            if len(slice_list) > 1:
                raise ValueError(
                    "Can't slice a dataset with multiple slices without key"
                )
            indexes = self.indexes[slice_list[0]]
            return DatasetView(
                dataset=self,
                indexes=indexes,
                lazy=self.lazy,
            )
        elif not slice_list:
            if subpath in self.keys:
                tensorview = TensorView(
                    dataset=self,
                    subpath=subpath,
                    slice_=slice(0, self._shape[0]),
                    lazy=self.lazy,
                )
                return tensorview if self.lazy else tensorview.compute()
            for key in self.keys:
                if subpath.startswith(key):
                    objectview = ObjectView(
                        dataset=self,
                        subpath=subpath,
                        lazy=self.lazy,
                        slice_=[slice(0, self._shape[0])],
                    )
                    return objectview if self.lazy else objectview.compute()
            return self._get_dictionary(subpath)
        else:
            schema_obj = self.schema.dict_[subpath.split("/")[1]]
            if subpath in self.keys and (
                not isinstance(schema_obj, Sequence) or len(slice_list) <= 1
            ):
                tensorview = TensorView(
                    dataset=self, subpath=subpath, slice_=slice_list, lazy=self.lazy
                )
                return tensorview if self.lazy else tensorview.compute()
            for key in self.keys:
                if subpath.startswith(key):
                    objectview = ObjectView(
                        dataset=self,
                        subpath=subpath,
                        slice_=slice_list,
                        lazy=self.lazy,
                    )
                    return objectview if self.lazy else objectview.compute()
            if len(slice_list) > 1:
                raise ValueError("You can't slice a dictionary of Tensors")
            return self._get_dictionary(subpath, slice_list[0])

    def __setitem__(self, slice_, value):
        """| Sets a slice or slices with a value
        | Usage:
        >>> ds["image", 5, 0:1920, 0:1080, 0:3] = np.zeros((1920, 1080, 3), "uint8")
        >>> images = ds["image"]
        >>> image = images[5]
        >>> image[0:1920, 0:1080, 0:3] = np.zeros((1920, 1080, 3), "uint8")
        """
        if "r" in self._mode:
            raise ReadModeException("__setitem__")
        self._auto_checkout()

        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)

        if not subpath:
            raise ValueError("Can't assign to dataset sliced without subpath")
        elif subpath not in self.keys:
            raise KeyError(f"Key {subpath} not found in the dataset")

        assign_value = get_value(value)
        schema_dict = self.schema
        if subpath[1:] in schema_dict.dict_.keys():
            schema_key = schema_dict.dict_.get(subpath[1:], None)
        else:
            for schema_key in subpath[1:].split("/"):
                schema_dict = schema_dict.dict_.get(schema_key, None)
                if not isinstance(schema_dict, SchemaDict):
                    schema_key = schema_dict
        if isinstance(schema_key, ClassLabel):
            assign_value = check_class_label(assign_value, schema_key)
        if isinstance(schema_key, (Text, bytes)) or (
            isinstance(assign_value, Iterable)
            and any(isinstance(val, str) for val in assign_value)
        ):
            # handling strings and bytes
            assign_value = str_to_int(assign_value, self.tokenizer)

        if not slice_list:
            self._tensors[subpath][:] = assign_value
        else:
            self._tensors[subpath][slice_list] = assign_value

    def filter(self, fn):
        """| Applies a function on each element one by one as a filter to get a new DatasetView

        Parameters
        ----------
        fn: function
            Should take in a single sample of the dataset and return True or False
            This function is applied to all the items of the datasetview and retains those items that return True
        """
        indexes = [index for index in self.indexes if fn(self[index])]
        return DatasetView(dataset=self, lazy=self.lazy, indexes=indexes)

    def store(
        self,
        url: str,
        token: dict = None,
        sample_per_shard: int = None,
        public: bool = True,
        scheduler="single",
        workers=1,
    ):
        """| Used to save the dataset as a new dataset, very similar to copy but uses transforms instead

        Parameters
        ----------
        url: str
            path where the data is going to be stored
        token: str or dict, optional
            If url is referring to a place where authorization is required,
            token is the parameter to pass the credentials, it can be filepath or dict
        length: int
            in case shape is None, user can provide length
        sample_per_shard: int
            How to split the iterator not to overfill RAM
        public: bool, optional
            only applicable if using hub storage, ignored otherwise
            setting this to False allows only the user who created it to access the dataset and
            the dataset won't be visible in the visualizer to the public
        scheduler: str
            choice between "single", "threaded", "processed"
        workers: int
            how many threads or processes to use
        Returns
        ----------
        ds: hub_v1.Dataset
            uploaded dataset
        """

        return _store_helper(
            self, url, token, sample_per_shard, public, scheduler, workers
        )

    def copy(self, dst_url: str, token=None, fs=None, public=True):
        """| Creates a copy of the dataset at the specified url and returns the dataset object
        Parameters
        ----------
        dst_url: str
            The destination url where dataset should be copied
        token: str or dict, optional
            If dst_url is refering to a place where authorization is required,
            token is the parameter to pass the credentials, it can be filepath or dict
        fs: optional
        public: bool, optional
            only applicable if using hub storage, ignored otherwise
            setting this to False allows only the user who created it to access the new copied dataset and
            the dataset won't be visible in the visualizer to the public
        """
        self.flush()
        destination = dst_url
        path = _copy_helper(
            dst_url=dst_url,
            token=token,
            fs=fs,
            public=public,
            src_url=self._path,
            src_fs=self._fs,
        )

        #  create entry in database if stored in hub storage
        if path.startswith("s3://snark-hub-dev/") or path.startswith("s3://snark-hub/"):
            subpath = path[5:]
            spl = subpath.split("/")
            if len(spl) < 4:
                raise ValueError("Invalid Path for dataset")
            username = spl[-2]
            dataset_name = spl[-1]
            HubControlClient().create_dataset_entry(
                username, dataset_name, self.meta, public=public
            )
        return hub_v1.Dataset(destination, token=token, fs=fs, public=public)

    def resize_shape(self, size: int) -> None:
        """Resize the shape of the dataset by resizing each tensor first dimension"""
        if size == self._shape[0]:
            return
        self._shape = (int(size),)
        self.indexes = list(range(self.shape[0]))
        self.meta = self._store_meta()
        for t in self._tensors.values():
            t.resize_shape(int(size))

        self._update_dataset_state()

    def append_shape(self, size: int):
        """Append the shape: Heavy Operation"""
        size += self._shape[0]
        self.resize_shape(size)

    def rename(self, name: str) -> None:
        """Renames the dataset"""
        self._name = name
        self.meta = self._store_meta()
        self.flush()

    def delete(self):
        """Deletes the dataset"""
        fs, path = self._fs, self._path
        exist_meta = fs.exists(posixpath.join(path, defaults.META_FILE))
        if exist_meta:
            fs.rm(path, recursive=True)
            if self.username:
                HubControlClient().delete_dataset_entry(
                    self.username, self.dataset_name
                )
            return True
        return False

    def to_pytorch(
        self,
        transform=None,
        inplace=True,
        output_type=dict,
        indexes=None,
        key_list=None,
    ):
        """| Converts the dataset into a pytorch compatible format.
        ** Pytorch does not support uint16, uint32, uint64 dtypes. These are implicitly type casted to int32, int64 and int64 respectively.
        Avoid having schema with these dtypes if you want to avoid this implicit conversion.
        ** This method does not work with Sequence schema

        Parameters
        ----------
        transform: function that transforms data in a dict format
        inplace: bool, optional
            Defines if data should be converted to torch.Tensor before or after Transforms applied (depends on what data
            type you need for Transforms). Default is True.
        output_type: one of list, tuple, dict, optional
            Defines the output type. Default is dict - same as in original Hub Dataset.
        indexes: list or int, optional
            The samples to be converted into Pytorch format. Takes all samples in dataset by default.
        key_list: list, optional
            The list of keys that are needed in Pytorch format. For nested schemas such as {"a":{"b":{"c": Tensor()}}}
            use ["a/b/c"] as key_list
        """
        from .integrations import _to_pytorch

        ds = _to_pytorch(self, transform, inplace, output_type, indexes, key_list)
        return ds

    def to_tensorflow(self, indexes=None, include_shapes=False, key_list=None):
        """| Converts the dataset into a tensorflow compatible format
        Parameters
        ----------
        indexes: list or int, optional
            The samples to be converted into tensorflow format. Takes all samples in dataset by default.
        include_shapes: boolean, optional
            False by default. Setting it to True passes the shapes to tf.data.Dataset.from_generator.
            Setting to True could lead to issues with dictionaries inside Tensors.
        key_list: list, optional
            The list of keys that are needed in tensorflow format. For nested schemas such as {"a":{"b":{"c": Tensor()}}}
            use ["a/b/c"] as key_list
        """
        from .integrations import _to_tensorflow

        ds = _to_tensorflow(self, indexes, include_shapes, key_list)
        return ds

    def _get_dictionary(self, subpath, slice_=None):
        """Gets dictionary from dataset given incomplete subpath"""
        tensor_dict = {}
        subpath = subpath if subpath.endswith("/") else subpath + "/"
        for key in self.keys:
            if key.startswith(subpath):
                suffix_key = key[len(subpath) :]
                split_key = suffix_key.split("/")
                cur = tensor_dict
                for i in range(len(split_key) - 1):
                    if split_key[i] not in cur.keys():
                        cur[split_key[i]] = {}
                    cur = cur[split_key[i]]
                slice_ = slice_ or slice(0, self._shape[0])
                tensorview = TensorView(
                    dataset=self, subpath=key, slice_=slice_, lazy=self.lazy
                )
                cur[split_key[-1]] = tensorview if self.lazy else tensorview.compute()
        if not tensor_dict:
            raise KeyError(f"Key {subpath} was not found in dataset")
        return tensor_dict

    def __iter__(self):
        """Returns Iterable over samples"""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """Number of samples in the dataset"""
        return self._shape[0]

    def disable_lazy(self):
        self.lazy = False

    def enable_lazy(self):
        self.lazy = True

    def _save_meta(self):
        _meta = json.loads(self._fs_map[defaults.META_FILE])
        _meta["meta_info"] = self._meta_information
        self._fs_map[defaults.META_FILE] = json.dumps(_meta).encode("utf-8")

    def flush(self):
        """Save changes from cache to dataset final storage. Doesn't create a new commit.
        Does not invalidate this object.
        """
        if "r" in self._mode:
            return
        self._store_version_info()
        for t in self._tensors.values():
            t.flush()
        self._save_meta()
        self._fs_map.flush()
        self._update_dataset_state()

    def save(self):
        """Save changes from cache to dataset final storage. Doesn't create a new commit.
        Does not invalidate this object.
        """
        self.flush()

    def close(self):
        """Save changes from cache to dataset final storage. Doesn't create a new commit.
        This invalidates this object.
        """
        self.flush()
        for t in self._tensors.values():
            t.close()
        self._fs_map.close()
        self._update_dataset_state()

    def _update_dataset_state(self):
        if self.username:
            HubControlClient().update_dataset_state(
                self.username, self.dataset_name, "UPLOADED"
            )

    def numpy(self, label_name=False):
        """Gets the values from different tensorview objects in the dataset schema

        Parameters
        ----------
        label_name: bool, optional
            If the TensorView object is of the ClassLabel type, setting this to True would retrieve the label names
            instead of the label encoded integers, otherwise this parameter is ignored.
        """
        return np.array(
            [
                create_numpy_dict(self, i, label_name=label_name)
                for i in range(self._shape[0])
            ]
        )

    def compute(self, label_name=False):
        """Gets the values from different tensorview objects in the dataset schema

        Parameters
        ----------
        label_name: bool, optional
            If the TensorView object is of the ClassLabel type, setting this to True would retrieve the label names
            instead of the label encoded integers, otherwise this parameter is ignored.
        """
        return self.numpy(label_name=label_name)

    def __str__(self):
        return (
            "Dataset(schema="
            + str(self._schema)
            + ", url="
            + "'"
            + self._url
            + "'"
            + ", shape="
            + str(self._shape)
            + ", mode="
            + "'"
            + self._mode
            + "')"
        )

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    @property
    def keys(self):
        """
        Get Keys of the dataset
        """
        return self._tensors.keys()

    @property
    def branches(self) -> list:
        """
        Gets a list all the branches of the dataset
        """
        if self._commit_id is None:
            raise VersioningNotSupportedException("branches")
        return self._branch_node_map.keys()

    def _get_mode(self, mode: str, fs: AbstractFileSystem):
        if mode:
            if mode not in ["r", "r+", "a", "a+", "w", "w+"]:
                raise Exception(f"Invalid mode {mode}")
            return mode
        else:
            try:
                meta_path = posixpath.join(self._path, defaults.META_FILE)
                if not fs.exists(self._path) or not fs.exists(meta_path):
                    return "a"
                bytes_ = bytes("Hello", "utf-8")
                path = posixpath.join(self._path, "mode_test")
                fs.pipe(path, bytes_)
                fs.rm(path)
            except BaseException:
                return "r"
            return "a"

    @staticmethod
    def from_tensorflow(ds, scheduler: str = "single", workers: int = 1):
        """Converts a tensorflow dataset into hub format.

        Parameters
        ----------
        dataset:
            The tensorflow dataset object that needs to be converted into hub format
        scheduler: str
            choice between "single", "threaded", "processed"
        workers: int
            how many threads or processes to use

        Examples
        --------
        >>> ds = tf.data.Dataset.from_tensor_slices(tf.range(10))
        >>> out_ds = hub_v1.Dataset.from_tensorflow(ds)
        >>> res_ds = out_ds.store("username/new_dataset") # res_ds is now a usable hub dataset

        >>> ds = tf.data.Dataset.from_tensor_slices({'a': [1, 2], 'b': [5, 6]})
        >>> out_ds = hub_v1.Dataset.from_tensorflow(ds)
        >>> res_ds = out_ds.store("username/new_dataset") # res_ds is now a usable hub dataset

        >>> ds = hub_v1.Dataset(schema=my_schema, shape=(1000,), url="username/dataset_name", mode="w")
        >>> ds = ds.to_tensorflow()
        >>> out_ds = hub_v1.Dataset.from_tensorflow(ds)
        >>> res_ds = out_ds.store("username/new_dataset") # res_ds is now a usable hub dataset
        """
        from .integrations import _from_tensorflow

        ds = _from_tensorflow(ds, scheduler, workers)
        return ds

    @staticmethod
    def from_tfds(
        dataset,
        split=None,
        num: int = -1,
        sampling_amount: int = 1,
        scheduler: str = "single",
        workers: int = 1,
    ):
        """| Converts a TFDS Dataset into hub format.

        Parameters
        ----------
        dataset: str
            The name of the tfds dataset that needs to be converted into hub format
        split: str, optional
            A string representing the splits of the dataset that are required such as "train" or "test+train"
            If not present, all the splits of the dataset are used.
        num: int, optional
            The number of samples required. If not present, all the samples are taken.
            If count is -1, or if count is greater than the size of this dataset, the new dataset will contain all elements of this dataset.
        sampling_amount: float, optional
            a value from 0 to 1, that specifies how much of the dataset would be sampled to determinte feature shapes
            value of 0 would mean no sampling and 1 would imply that entire dataset would be sampled
        scheduler: str
            choice between "single", "threaded", "processed"
        workers: int
            how many threads or processes to use

        Examples
        --------
        >>> out_ds = hub_v1.Dataset.from_tfds('mnist', split='test+train', num=1000)
        >>> res_ds = out_ds.store("username/mnist") # res_ds is now a usable hub dataset
        """
        from .integrations import _from_tfds

        ds = _from_tfds(dataset, split, num, sampling_amount, scheduler, workers)
        return ds

    @staticmethod
    def from_pytorch(dataset, scheduler: str = "single", workers: int = 1):
        """| Converts a pytorch dataset object into hub format

        Parameters
        ----------
        dataset:
            The pytorch dataset object that needs to be converted into hub format
        scheduler: str
            choice between "single", "threaded", "processed"
        workers: int
            how many threads or processes to use
        """

        from .integrations import _from_pytorch

        ds = _from_pytorch(dataset, scheduler, workers)
        return ds

    @staticmethod
    def from_path(path, scheduler="single", workers=1):
        # infer schema & get data (label -> input mapping with file refs)
        ds = auto.infer_dataset(path, scheduler=scheduler, workers=workers)
        return ds
