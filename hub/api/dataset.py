from typing import Tuple
import posixpath
import collections.abc as abc
import json
import sys
import os
import traceback

import fsspec
import numcodecs
import numcodecs.lz4
import numcodecs.zstd

from hub.features.features import (
    Primitive,
    Tensor,
    FeatureDict,
    HubFeature,
    featurify,
    FlatTensor,
)
from hub.log import logger

from hub.api.tensorview import TensorView
from hub.api.datasetview import DatasetView
from hub.api.dataset_utils import slice_extract_info, slice_split
from hub.utils import compute_lcm

import hub.features.serialize
import hub.features.deserialize
from hub.features.features import flatten

from hub.store.dynamic_tensor import DynamicTensor
from hub.store.store import get_fs_and_path, get_storage_map
from hub.exceptions import (
    HubDatasetNotFoundException,
    NotHubDatasetToOverwriteException,
    NotHubDatasetToAppendException,
    ShapeArgumentNotFoundException,
    SchemaArgumentNotFoundException,
    ModuleNotInstalledException,
    WrongUsernameException,
)
from hub.store.metastore import MetaStorage
from hub.client.hub_control import HubControlClient
from hub.features.image import Image
from hub.features.class_label import ClassLabel
import traceback

try:
    import torch
except ImportError:
    pass
try:
    import tensorflow as tf
except ImportError:
    pass
try:
    import tensorflow_datasets as tfds
except ImportError:
    pass


def get_file_count(fs: fsspec.AbstractFileSystem, path):
    return len(fs.listdir(path, detail=False))


class Dataset:
    def __init__(
        self,
        url: str,
        mode: str = "a",
        safe_mode: bool = False,
        shape=None,
        schema=None,
        token=None,
        fs=None,
        fs_map=None,
        cache: int = 2 ** 26,
        lock_cache=True,
    ):
        """Open a new or existing dataset for read/write

        Parameters
        ----------
        url: str
            The url where dataset is located/should be created
        mode: str, optional (default to "w")
            Python way to tell whether dataset is for read or write (ex. "r", "w", "a")
        safe_mode: bool, optional
            if dataset exists it cannot be rewritten in safe mode, otherwise it lets to write the first time
        shape: tuple, optional
            Tuple with (num_samples,) format, where num_samples is number of samples
        schema: optional
            Describes the data of a single sample. Hub features are used for that
            Required for 'a' and 'w' modes
        token: str or dict, optional
            If url is refering to a place where authorization is required,
            token is the parameter to pass the credentials, it can be filepath or dict
        fs: optional
        fs_map: optional
        cache: int, optional
            Size of the cache. Default is 2GB (2**20)
        lock_cache: bool, optional
            Lock the cache for avoiding multiprocessing errors
        """

        shape = shape or (None,)
        if isinstance(shape, int):
            shape = [shape]
        if shape is not None:
            assert len(tuple(shape)) == 1
        assert mode is not None

        self.url = url
        self.token = token
        self.mode = mode

        self._fs, self._path = (
            (fs, url) if fs else get_fs_and_path(self.url, token=token)
        )
        self.cache = cache
        self.lock_cache = lock_cache

        needcreate = self._check_and_prepare_dir()
        fs_map = fs_map or get_storage_map(self._fs, self._path, cache, lock=lock_cache)
        self._fs_map = fs_map

        if safe_mode and not needcreate:
            mode = "r"

        if not needcreate:
            meta = json.loads(fs_map["meta.json"].decode("utf-8"))
            self.shape = tuple(meta["shape"])
            self.schema = hub.features.deserialize.deserialize(meta["schema"])
            self._flat_tensors = tuple(flatten(self.schema))
            self._tensors = dict(self._open_storage_tensors())
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
                self.schema: HubFeature = featurify(schema)
                self.shape = tuple(shape)
                self.meta = {
                    "shape": shape,
                    "schema": hub.features.serialize.serialize(self.schema),
                    "version": 1,
                }
                fs_map["meta.json"] = bytes(json.dumps(self.meta), "utf-8")
                self._flat_tensors = tuple(flatten(self.schema))
                self._tensors = dict(self._generate_storage_tensors())
            except Exception as e:
                self._fs.rm(self._path, recursive=True)
                logger.error("Deleting the dataset " + traceback.format_exc() + str(e))
                raise

        self.username = None
        self.dataset_name = None
        if self._path.startswith("s3://snark-hub-dev/") or self._path.startswith(
            "s3://snark-hub/"
        ):
            subpath = self._path[5:]
            spl = subpath.split("/")
            if len(spl) < 4:
                raise ValueError("Invalid Path for dataset")
            self.username = spl[-2]
            self.dataset_name = spl[-1]
            HubControlClient().create_dataset_entry(
                self.username, self.dataset_name, self.meta
            )

    def _check_and_prepare_dir(self):
        """
        Checks if input data is ok.
        Creates or overwrites dataset folder.
        Returns True dataset needs to be created opposed to read.
        """
        fs, path, mode = self._fs, self._path, self.mode
        if path.startswith("s3://"):
            with open(os.path.expanduser("~/.activeloop/store"), "rb") as f:
                stored_username = json.load(f)["_id"]
            current_username = path.split("/")[-2]
            if stored_username != current_username:
                raise WrongUsernameException(current_username)
        exist_meta = fs.exists(posixpath.join(path, "meta.json"))
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

    def _get_dynamic_tensor_dtype(self, t_dtype):
        if isinstance(t_dtype, Primitive):
            return t_dtype.dtype
        elif isinstance(t_dtype.dtype, Primitive):
            return t_dtype.dtype.dtype
        else:
            return "object"

    def _get_compressor(self, compressor: str):
        if compressor.lower() == "lz4":
            return numcodecs.LZ4(numcodecs.lz4.DEFAULT_ACCELERATION)
        elif compressor.lower() == "zstd":
            return numcodecs.Zstd(numcodecs.zstd.DEFAULT_CLEVEL)
        else:
            raise ValueError(
                f"Wrong compressor: {compressor}, only LZ4 and ZSTD are supported"
            )

    def _generate_storage_tensors(self):
        for t in self._flat_tensors:
            t_dtype, t_path = t
            path = posixpath.join(self._path, t_path[1:])
            self._fs.makedirs(posixpath.join(path, "--dynamic--"))
            yield t_path, DynamicTensor(
                fs_map=MetaStorage(
                    t_path,
                    get_storage_map(self._fs, path, self.cache, self.lock_cache),
                    self._fs_map,
                ),
                mode=self.mode,
                shape=self.shape + t_dtype.shape,
                max_shape=self.shape + t_dtype.max_shape,
                dtype=self._get_dynamic_tensor_dtype(t_dtype),
                chunks=t_dtype.chunks,
                compressor=self._get_compressor(t_dtype.compressor),
            )

    def _open_storage_tensors(self):
        for t in self._flat_tensors:
            t_dtype, t_path = t
            path = posixpath.join(self._path, t_path[1:])
            yield t_path, DynamicTensor(
                fs_map=MetaStorage(
                    t_path,
                    get_storage_map(self._fs, path, self.cache, self.lock_cache),
                    self._fs_map,
                ),
                mode=self.mode,
                # FIXME We don't need argument below here
                shape=self.shape + t_dtype.shape,
            )

    def __getitem__(self, slice_):
        """| Gets a slice or slices from dataset
        | Usage:

        >>> return ds["image", 5, 0:1920, 0:1080, 0:3].numpy() # returns numpy array

        >>> images = ds["image"]
        >>> return images[5].numpy() # returns numpy array

        >>> images = ds["image"]
        >>> image = images[5]
        >>> return image[0:1920, 0:1080, 0:3].numpy()
        """
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        if not subpath:
            if len(slice_list) > 1:
                raise ValueError(
                    "Can't slice a dataset with multiple slices without subpath"
                )
            num, ofs = slice_extract_info(slice_list[0], self.shape[0])
            return DatasetView(dataset=self, num_samples=num, offset=ofs)
        elif not slice_list:
            if subpath in self._tensors.keys():
                return TensorView(
                    dataset=self, subpath=subpath, slice_=slice(0, self.shape[0])
                )
            return self._get_dictionary(subpath)
        else:
            num, ofs = slice_extract_info(slice_list[0], self.shape[0])
            if subpath in self._tensors.keys():
                return TensorView(dataset=self, subpath=subpath, slice_=slice_list)
            if len(slice_list) > 1:
                raise ValueError("You can't slice a dictionary of Tensors")
            return self._get_dictionary(subpath, slice_list[0])

    def __setitem__(self, slice_, value):
        """| Sets a slice or slices with a value
        | Usage

         >>> ds["image", 5, 0:1920, 0:1080, 0:3] = np.zeros((1920, 1080, 3), "uint8")

        >>> images = ds["image"]
        >>> image = images[5]
        >>> image[0:1920, 0:1080, 0:3] = np.zeros((1920, 1080, 3), "uint8")
        """
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        if not subpath:
            raise ValueError("Can't assign to dataset sliced without subpath")
        elif not slice_list:
            self._tensors[subpath][:] = value  # Add path check
        else:
            self._tensors[subpath][slice_list] = value

    def delete(self):
        fs, path = self._fs, self._path
        exist_meta = fs.exists(posixpath.join(path, "meta.json"))
        if exist_meta:
            fs.rm(path, recursive=True)
            if self.username is not None:
                HubControlClient().delete_dataset_entry(
                    self.username, self.dataset_name
                )
            return True
        return False

    def to_pytorch(self, Transform=None):
        """Converts the dataset into a pytorch compatible format"""
        if "torch" not in sys.modules:
            raise ModuleNotInstalledException("torch")
        return TorchDataset(self, Transform)

    def to_tensorflow(self):
        """Converts the dataset into a tensorflow compatible format"""
        if "tensorflow" not in sys.modules:
            raise ModuleNotInstalledException("tensorflow")

        def tf_gen():
            for index in range(self.shape[0]):
                d = {}
                for key in self._tensors.keys():
                    split_key = key.split("/")
                    cur = d
                    for i in range(1, len(split_key) - 1):
                        if split_key[i] in cur.keys():
                            cur = cur[split_key[i]]
                        else:
                            cur[split_key[i]] = {}
                            cur = cur[split_key[i]]
                    cur[split_key[-1]] = self._tensors[key][index]
                yield (d)

        def dict_to_tf(my_dtype):
            d = {}
            for k, v in my_dtype.dict_.items():
                d[k] = dtype_to_tf(v)
            return d

        def tensor_to_tf(my_dtype):
            return dtype_to_tf(my_dtype.dtype)

        def dtype_to_tf(my_dtype):
            if isinstance(my_dtype, FeatureDict):
                return dict_to_tf(my_dtype)
            elif isinstance(my_dtype, Tensor):
                return tensor_to_tf(my_dtype)
            elif isinstance(my_dtype, Primitive):
                return str(my_dtype._dtype)

        def get_output_shapes(my_dtype):
            if isinstance(my_dtype, FeatureDict):
                return output_shapes_from_dict(my_dtype)
            elif isinstance(my_dtype, Tensor):
                return my_dtype.shape
            elif isinstance(my_dtype, Primitive):
                return ()

        def output_shapes_from_dict(my_dtype):
            d = {}
            for k, v in my_dtype.dict_.items():
                d[k] = get_output_shapes(v)
            return d

        output_types = dtype_to_tf(self.schema)
        output_shapes = get_output_shapes(self.schema)
        # print(output_shapes)
        return tf.data.Dataset.from_generator(
            tf_gen, output_types=output_types, output_shapes=output_shapes
        )

    def _get_dictionary(self, subpath, slice_=None):
        """"Gets dictionary from dataset given incomplete subpath"""
        tensor_dict = {}
        subpath = subpath if subpath.endswith("/") else subpath + "/"
        for key in self._tensors.keys():
            if key.startswith(subpath):
                suffix_key = key[len(subpath) :]
                split_key = suffix_key.split("/")
                cur = tensor_dict
                for i in range(len(split_key) - 1):
                    if split_key[i] not in cur.keys():
                        cur[split_key[i]] = {}
                    cur = cur[split_key[i]]
                slice_ = slice_ if slice_ else slice(0, self.shape[0])
                cur[split_key[-1]] = TensorView(
                    dataset=self, subpath=key, slice_=slice_
                )
        if len(tensor_dict) == 0:
            raise KeyError(f"Key {subpath} was not found in dataset")
        return tensor_dict

    def __iter__(self):
        """ Returns Iterable over samples """
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """ Number of samples in the dataset """
        return self.shape[0]

    def commit(self):
        """Save changes from cache to dataset final storage
        This invalidates this object
        """
        for t in self._tensors.values():
            t.commit()
        if self.username is not None:
            HubControlClient().update_dataset_state(
                self.username, self.dataset_name, "UPLOADED"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.commit()

    @property
    def chunksize(self):
        # FIXME assumes chunking is done on the first sample
        chunks = [t.chunksize[0] for t in self._tensors.values()]
        return compute_lcm(chunks)

    @property
    def keys(self):
        """
        Get Keys of the dataset
        """
        return self._tensors.keys()

    @staticmethod
    def from_tensorflow(ds):
        """Converts a tensorflow dataset into hub format
        Parameters
        ----------
        dataset:
            The tensorflow dataset object that needs to be converted into hub format
        Examples
        --------
        ds = tf.data.Dataset.from_tensor_slices(tf.range(10))
        out_ds = hub.Dataset.from_tensorflow(ds)
        res_ds = out_ds.store("username/new_dataset") # res_ds is now a usable hub dataset

        ds = tf.data.Dataset.from_tensor_slices({'a': [1, 2], 'b': [5, 6]})
        out_ds = hub.Dataset.from_tensorflow(ds)
        res_ds = out_ds.store("username/new_dataset") # res_ds is now a usable hub dataset

        ds = hub.Dataset(schema=my_schema, shape=(1000,), url="username/dataset_name", mode="w")
        ds = ds.to_tensorflow()
        out_ds = hub.Dataset.from_tensorflow(ds)
        res_ds = out_ds.store("username/new_dataset") # res_ds is now a usable hub dataset

        """
        if "tensorflow" not in sys.modules:
            raise ModuleNotInstalledException("tensorflow")

        def generate_schema(ds):
            if isinstance(ds._structure, tf.python.framework.tensor_spec.TensorSpec):
                return tf_to_hub({"data": ds._structure}).dict_
            return tf_to_hub(ds._structure).dict_

        def tf_to_hub(tf_dt):
            if isinstance(tf_dt, dict):
                return dict_to_hub(tf_dt)
            elif isinstance(tf_dt, tf.python.framework.tensor_spec.TensorSpec):
                return TensorSpec_to_hub(tf_dt)

        def TensorSpec_to_hub(tf_dt):
            shape = tf_dt.shape if tf_dt.shape.rank is not None else (None,)
            return Tensor(shape=shape, dtype=tf_dt.dtype.name)

        def dict_to_hub(tf_dt):
            d = {
                key.replace("/", "_"): tf_to_hub(value) for key, value in tf_dt.items()
            }
            return FeatureDict(d)

        my_schema = generate_schema(ds)

        def transform_numpy(sample):
            d = {}
            for k, v in sample.items():
                k = k.replace("/", "_")
                if not isinstance(v, dict):
                    if isinstance(v, tuple) or isinstance(v, list):
                        new_v = list(v)
                        for i in range(len(new_v)):
                            new_v[i] = new_v[i].numpy()
                        d[k] = tuple(new_v) if isinstance(v, tuple) else new_v
                    else:
                        d[k] = v.numpy()
                else:
                    d[k] = transform_numpy(v)
            return d

        @hub.transform(schema=my_schema)
        def my_transform(sample):
            sample = sample if isinstance(sample, dict) else {"data": sample}
            return transform_numpy(sample)

        return my_transform(ds)

    @staticmethod
    def from_tfds(dataset, split=None, num=-1):
        """Converts a TFDS Dataset into hub format
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
        Examples
        --------
        out_ds = hub.Dataset.from_tfds('mnist', split='test+train', num=1000)
        res_ds = out_ds.store("username/mnist") # res_ds is now a usable hub dataset
        """
        if "tensorflow_datasets" not in sys.modules:
            raise ModuleNotInstalledException("tensorflow_datasets")
        ds_info = tfds.load(dataset, with_info=True)
        if split is None:
            all_splits = ds_info[1].splits.keys()
            split = "+".join(all_splits)
        ds = tfds.load(dataset, split=split)
        ds = ds.take(num)

        def generate_schema(ds):
            tf_schema = ds[1].features
            schema = to_hub(tf_schema).dict_
            return schema

        def to_hub(tf_dt):
            if isinstance(tf_dt, tfds.features.FeaturesDict):
                return fdict_to_hub(tf_dt)
            elif isinstance(tf_dt, tfds.features.Tensor):
                return tensor_to_hub(tf_dt)
            elif isinstance(tf_dt, tfds.features.Image):
                return image_to_hub(tf_dt)
            elif isinstance(tf_dt, tfds.features.ClassLabel):
                return class_label_to_hub(tf_dt)
            elif isinstance(tf_dt, tfds.features.Text):
                return text_to_hub(tf_dt)
            elif isinstance(tf_dt, tfds.features.Sequence):
                return sequence_to_hub(tf_dt)
            else:
                if tf_dt.dtype.name != "string":
                    return tf_dt.dtype.name

        def fdict_to_hub(tf_dt):
            d = {key.replace("/", "_"): to_hub(value) for key, value in tf_dt.items()}
            return FeatureDict(d)

        def tensor_to_hub(tf_dt):
            dt = tf_dt.dtype.name if tf_dt.dtype.name != "string" else "object"
            max_shape = tuple(10000 if dim is None else dim for dim in tf_dt.shape)
            return Tensor(shape=tf_dt.shape, dtype=dt, max_shape=max_shape)

        def image_to_hub(tf_dt):
            dt = tf_dt.dtype.name if tf_dt.dtype.name != "string" else "object"
            max_shape = tuple(10000 if dim is None else dim for dim in tf_dt.shape)
            return Image(shape=tf_dt.shape, dtype=dt, max_shape=max_shape)

        def class_label_to_hub(tf_dt):
            dt = tf_dt.dtype.name if tf_dt.dtype.name != "string" else "object"
            max_shape = tuple(10000 if dim is None else dim for dim in tf_dt.shape)
            if hasattr(tf_dt, "_num_classes"):
                return ClassLabel(
                    shape=tf_dt.shape,
                    dtype=dt,
                    num_classes=tf_dt.num_classes,
                    max_shape=max_shape,
                )
            else:
                return ClassLabel(
                    shape=tf_dt.shape, dtype=dt, names=tf_dt.names, max_shape=max_shape
                )

        def text_to_hub(tf_dt):
            max_shape = tuple(10000 if dim is None else dim for dim in tf_dt.shape)
            dt = tf_dt.dtype.name if tf_dt.dtype.name != "string" else "object"
            return Tensor(shape=tf_dt.shape, dtype=dt, max_shape=max_shape)

        def sequence_to_hub(tf_dt):
            return "object"

        my_schema = generate_schema(ds_info)

        def transform_numpy(sample):
            d = {}
            for k, v in sample.items():
                k = k.replace("/", "_")
                if not isinstance(v, dict):
                    d[k] = v.numpy()
                else:
                    d[k] = transform_numpy(v)
            return d

        @hub.transform(schema=my_schema)
        def my_transform(sample):
            return transform_numpy(sample)

        return my_transform(ds)


class TorchDataset:
    def __init__(self, ds, transform=None):
        self._ds = None
        self._url = ds.url
        self._token = ds.token
        self._transform = transform

    def _do_transform(self, data):
        return self._transform(data) if self._transform else data

    def _init_ds(self):
        """
        For each process, dataset should be independently loaded
        """
        if self._ds is None:
            self._ds = Dataset(self._url, token=self._token, lock_cache=False)

    def __len__(self):
        self._init_ds()
        return self._ds.shape[0]

    def __getitem__(self, index):
        self._init_ds()
        d = {}
        for key in self._ds._tensors.keys():
            split_key = key.split("/")
            cur = d
            for i in range(1, len(split_key) - 1):
                if split_key[i] in cur.keys():
                    cur = cur[split_key[i]]
                else:
                    cur[split_key[i]] = {}
                    cur = cur[split_key[i]]

            cur[split_key[-1]] = torch.tensor(self._ds._tensors[key][index])
        return d

    def __iter__(self):
        self._init_ds()
        for index in range(self.shape[0]):
            d = {}
            for key in self._ds._tensors.keys():
                split_key = key.split("/")
                cur = d
                for i in range(1, len(split_key) - 1):
                    if split_key[i] in cur.keys():
                        cur = cur[split_key[i]]
                    else:
                        cur[split_key[i]] = {}
                        cur = cur[split_key[i]]
                cur[split_key[-1]] = torch.tensor(self._tensors[key][index])
            yield (d)
