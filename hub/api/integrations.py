"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import sys
import numpy as np
import json
from itertools import chain
from collections import defaultdict
import PIL.Image
import PIL.ImageDraw
from hub.exceptions import ModuleNotInstalledException, OutOfBoundsError
from hub.schema.features import Primitive, Tensor, SchemaDict
from hub.schema import Audio, BBox, ClassLabel, Image, Sequence, Text, Video, Mask
from .dataset import Dataset
import hub.store.pickle_s3_storage
import hub.schema.serialize
import hub.schema.deserialize
import random


def _to_pytorch(
    dataset,
    transform=None,
    inplace=True,
    output_type=dict,
    indexes=None,
    key_list=None,
    shuffle=False,
):
    """| Converts the dataset into a pytorch compatible format.

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
    shuffle: bool, optional
        whether to shuffle the data chunkwise or not. Default is False.
    """
    try:
        import torch
    except ModuleNotFoundError:
        raise ModuleNotInstalledException("torch")

    global torch
    indexes = indexes or dataset.indexes

    if "r" not in dataset.mode:
        dataset.flush()  # FIXME Without this some tests in test_converters.py fails, not clear why
    return TorchDataset(
        dataset,
        transform,
        inplace=inplace,
        output_type=output_type,
        indexes=indexes,
        key_list=key_list,
        shuffle=shuffle,
    )


def _from_pytorch(dataset, scheduler: str = "single", workers: int = 1):
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

    if "torch" not in sys.modules:
        raise ModuleNotInstalledException("torch")
    else:
        import torch

        global torch

    max_dict = defaultdict(lambda: None)

    def sampling(ds):
        for sample in ds:
            dict_sampling(sample)

    def dict_sampling(d, path=""):
        for k, v in d.items():
            k = k.replace("/", "_")
            cur_path = path + "/" + k
            if isinstance(v, dict):
                dict_sampling(v, path=cur_path)
            elif isinstance(v, str):
                if cur_path not in max_dict.keys():
                    max_dict[cur_path] = (len(v),)
                else:
                    max_dict[cur_path] = max(((len(v)),), max_dict[cur_path])
            elif hasattr(v, "shape"):
                if cur_path not in max_dict.keys():
                    max_dict[cur_path] = v.shape
                else:
                    max_dict[cur_path] = tuple(
                        [max(value) for value in zip(max_dict[cur_path], v.shape)]
                    )

    sampling(dataset)

    def generate_schema(dataset):
        sample = dataset[0]
        return dict_to_hub(sample).dict_

    def dict_to_hub(dic, path=""):
        d = {}
        for k, v in dic.items():
            k = k.replace("/", "_")
            cur_path = path + "/" + k
            if isinstance(v, dict):
                d[k] = dict_to_hub(v, path=cur_path)
            else:
                value_shape = v.shape if hasattr(v, "shape") else ()
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                shape = tuple(None for it in value_shape)
                max_shape = (
                    max_dict[cur_path] or tuple(10000 for it in value_shape)
                    if not isinstance(v, str)
                    else (10000,)
                )
                dtype = v.dtype.name if hasattr(v, "dtype") else type(v)
                dtype = "int64" if isinstance(v, str) else dtype
                d[k] = (
                    Tensor(shape=shape, dtype=dtype, max_shape=max_shape)
                    if not isinstance(v, str)
                    else Text(shape=(None,), dtype=dtype, max_shape=max_shape)
                )
        return SchemaDict(d)

    my_schema = generate_schema(dataset)

    def transform_numpy(sample):
        d = {}
        for k, v in sample.items():
            k = k.replace("/", "_")
            d[k] = transform_numpy(v) if isinstance(v, dict) else v
        return d

    @hub.transform(schema=my_schema, scheduler=scheduler, workers=workers)
    def my_transform(sample):
        return transform_numpy(sample)

    return my_transform(dataset)


def _to_tensorflow(dataset, indexes=None, include_shapes=False, key_list=None):
    """| Converts the dataset into a tensorflow compatible format

    Parameters
    ----------
    indexes: list or int, optional
        The samples to be converted into tensorflow format. Takes all samples in dataset by default.
    include_shapes: boolean, optional
        False by default. Setting it to True passes the shapes to tf.data.Dataset.from_generator.
        Setting to True could lead to issues with dictionaries inside Tensors.
    """
    try:
        import tensorflow as tf

        global tf
    except ModuleNotFoundError:
        raise ModuleNotInstalledException("tensorflow")
    key_list = key_list or list(dataset.keys)
    key_list = [key if key.startswith("/") else "/" + key for key in key_list]
    for key in key_list:
        if key not in dataset.keys:
            raise KeyError(key)
    indexes = indexes or dataset.indexes
    indexes = [indexes] if isinstance(indexes, int) else indexes
    _samples_in_chunks = {
        key: value.chunks[0] for key, value in dataset._tensors.items()
    }
    _active_chunks = {}
    _active_chunks_range = {}

    def _get_active_item(key, index):
        active_range = _active_chunks_range.get(key)
        samples_per_chunk = _samples_in_chunks[key]
        if active_range is None or index not in active_range:
            active_range_start = index - index % samples_per_chunk
            active_range = range(
                active_range_start,
                min(active_range_start + samples_per_chunk, indexes[-1] + 1),
            )
            _active_chunks_range[key] = active_range
            _active_chunks[key] = dataset._tensors[key][
                active_range.start : active_range.stop
            ]
        return _active_chunks[key][index % samples_per_chunk]

    def tf_gen():
        key_dtype_map = {key: dataset[key, indexes[0]].dtype for key in dataset.keys}
        for index in indexes:
            d = {}
            for key in dataset.keys:
                if key not in key_list:
                    continue
                split_key, cur = key.split("/"), d
                for i in range(1, len(split_key) - 1):
                    if split_key[i] in cur.keys():
                        cur = cur[split_key[i]]
                    else:
                        cur[split_key[i]] = {}
                        cur = cur[split_key[i]]
                cur[split_key[-1]] = _get_active_item(key, index)
                if isinstance(key_dtype_map[key], Text):
                    value = cur[split_key[-1]]
                    cur[split_key[-1]] = (
                        "".join(chr(it) for it in value.tolist())
                        if value.ndim == 1
                        else ["".join(chr(it) for it in val.tolist()) for val in value]
                    )

            yield (d)

    def dict_to_tf(my_dtype, path=""):
        d = {}
        for k, v in my_dtype.dict_.items():
            for key in key_list:
                if key.startswith(path + "/" + k):
                    d[k] = dtype_to_tf(v, path + "/" + k)
                    break
        return d

    def tensor_to_tf(my_dtype):
        return dtype_to_tf(my_dtype.dtype)

    def text_to_tf(my_dtype):
        return "string"

    def dtype_to_tf(my_dtype, path=""):
        if isinstance(my_dtype, SchemaDict):
            return dict_to_tf(my_dtype, path=path)
        elif isinstance(my_dtype, Text):
            return text_to_tf(my_dtype)
        elif isinstance(my_dtype, Tensor):
            return tensor_to_tf(my_dtype)
        elif isinstance(my_dtype, Primitive):
            if str(my_dtype._dtype) == "object":
                return "string"
            return str(my_dtype._dtype)

    def get_output_shapes(my_dtype, path=""):
        if isinstance(my_dtype, SchemaDict):
            return output_shapes_from_dict(my_dtype, path=path)
        elif isinstance(my_dtype, (Text, Primitive)):
            return ()
        elif isinstance(my_dtype, Tensor):
            return my_dtype.shape

    def output_shapes_from_dict(my_dtype, path=""):
        d = {}
        for k, v in my_dtype.dict_.items():
            for key in key_list:
                if key.startswith(path + "/" + k):
                    d[k] = get_output_shapes(v, path + "/" + k)
                    break
        return d

    output_types = dtype_to_tf(dataset._schema)
    if include_shapes:
        output_shapes = get_output_shapes(dataset._schema)
        return tf.data.Dataset.from_generator(
            tf_gen, output_types=output_types, output_shapes=output_shapes
        )
    else:
        return tf.data.Dataset.from_generator(tf_gen, output_types=output_types)


def _from_tensorflow(ds, scheduler: str = "single", workers: int = 1):
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
    >>> out_ds = hub.Dataset.from_tensorflow(ds)
    >>> res_ds = out_ds.store("username/new_dataset") # res_ds is now a usable hub dataset

    >>> ds = tf.data.Dataset.from_tensor_slices({'a': [1, 2], 'b': [5, 6]})
    >>> out_ds = hub.Dataset.from_tensorflow(ds)
    >>> res_ds = out_ds.store("username/new_dataset") # res_ds is now a usable hub dataset

    >>> ds = hub.Dataset(schema=my_schema, shape=(1000,), url="username/dataset_name", mode="w")
    >>> ds = ds.to_tensorflow()
    >>> out_ds = hub.Dataset.from_tensorflow(ds)
    >>> res_ds = out_ds.store("username/new_dataset") # res_ds is now a usable hub dataset
    """
    if "tensorflow" not in sys.modules:
        raise ModuleNotInstalledException("tensorflow")
    else:
        import tensorflow as tf

        global tf

    def generate_schema(ds):
        if isinstance(ds._structure, tf.TensorSpec):
            return tf_to_hub({"data": ds._structure}).dict_
        return tf_to_hub(ds._structure).dict_

    def tf_to_hub(tf_dt):
        if isinstance(tf_dt, dict):
            return dict_to_hub(tf_dt)
        elif isinstance(tf_dt, tf.TensorSpec):
            return TensorSpec_to_hub(tf_dt)

    def TensorSpec_to_hub(tf_dt):
        dt = tf_dt.dtype.name if tf_dt.dtype.name != "string" else "object"
        shape = tuple(tf_dt.shape) if tf_dt.shape.rank is not None else (None,)
        return Tensor(shape=shape, dtype=dt)

    def dict_to_hub(tf_dt):
        d = {key.replace("/", "_"): tf_to_hub(value) for key, value in tf_dt.items()}
        return SchemaDict(d)

    my_schema = generate_schema(ds)

    def transform_numpy(sample):
        d = {}
        for k, v in sample.items():
            k = k.replace("/", "_")
            if not isinstance(v, dict):
                if isinstance(v, (tuple, list)):
                    new_v = list(v)
                    for i in range(len(new_v)):
                        new_v[i] = new_v[i].numpy()
                    d[k] = tuple(new_v) if isinstance(v, tuple) else new_v
                else:
                    d[k] = v.numpy()
            else:
                d[k] = transform_numpy(v)
        return d

    @hub.transform(schema=my_schema, scheduler=scheduler, workers=workers)
    def my_transform(sample):
        sample = sample if isinstance(sample, dict) else {"data": sample}
        return transform_numpy(sample)

    return my_transform(ds)


def _from_tfds(
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
    >>> out_ds = hub.Dataset.from_tfds('mnist', split='test+train', num=1000)
    >>> res_ds = out_ds.store("username/mnist") # res_ds is now a usable hub dataset
    """
    try:
        import tensorflow_datasets as tfds

        global tfds
    except Exception:
        raise ModuleNotInstalledException("tensorflow_datasets")

    ds_info = tfds.load(dataset, with_info=True)

    if split is None:
        all_splits = ds_info[1].splits.keys()
        split = "+".join(all_splits)

    ds = tfds.load(dataset, split=split)
    ds = ds.take(num)
    max_dict = defaultdict(lambda: None)

    def sampling(ds):
        try:
            subset_len = len(ds) if hasattr(ds, "__len__") else num
        except Exception:
            subset_len = max(num, 5)

        subset_len = int(max(subset_len * sampling_amount, 5))
        samples = ds.take(subset_len)
        for smp in samples:
            dict_sampling(smp)

    def dict_sampling(d, path=""):
        for k, v in d.items():
            k = k.replace("/", "_")
            cur_path = path + "/" + k
            if isinstance(v, dict):
                dict_sampling(v)
            elif hasattr(v, "shape") and v.dtype != "string":
                if cur_path not in max_dict.keys():
                    max_dict[cur_path] = v.shape
                else:
                    max_dict[cur_path] = tuple(
                        [max(value) for value in zip(max_dict[cur_path], v.shape)]
                    )
            elif hasattr(v, "shape") and v.dtype == "string":
                if cur_path not in max_dict.keys():
                    max_dict[cur_path] = (len(v.numpy()),)
                else:
                    max_dict[cur_path] = max(((len(v.numpy()),), max_dict[cur_path]))

    if sampling_amount > 0:
        sampling(ds)

    def generate_schema(ds):
        tf_schema = ds[1].features
        return to_hub(tf_schema).dict_

    def to_hub(tf_dt, max_shape=None, path=""):
        if isinstance(tf_dt, tfds.features.FeaturesDict):
            return sdict_to_hub(tf_dt, path=path)
        elif isinstance(tf_dt, tfds.features.Image):
            return image_to_hub(tf_dt, max_shape=max_shape)
        elif isinstance(tf_dt, tfds.features.ClassLabel):
            return class_label_to_hub(tf_dt, max_shape=max_shape)
        elif isinstance(tf_dt, tfds.features.Video):
            return video_to_hub(tf_dt, max_shape=max_shape)
        elif isinstance(tf_dt, tfds.features.Text):
            return text_to_hub(tf_dt, max_shape=max_shape)
        elif isinstance(tf_dt, tfds.features.Sequence):
            return sequence_to_hub(tf_dt, max_shape=max_shape)
        elif isinstance(tf_dt, tfds.features.BBoxFeature):
            return bbox_to_hub(tf_dt, max_shape=max_shape)
        elif isinstance(tf_dt, tfds.features.Audio):
            return audio_to_hub(tf_dt, max_shape=max_shape)
        elif isinstance(tf_dt, tfds.features.Tensor):
            return tensor_to_hub(tf_dt, max_shape=max_shape)
        else:
            if tf_dt.dtype.name != "string":
                return tf_dt.dtype.name

    def sdict_to_hub(tf_dt, path=""):
        d = {}
        for key, value in tf_dt.items():
            key = key.replace("/", "_")
            cur_path = path + "/" + key
            d[key] = to_hub(value, max_dict[cur_path], cur_path)
        return SchemaDict(d)

    def tensor_to_hub(tf_dt, max_shape=None):
        if tf_dt.dtype.name == "string":
            max_shape = max_shape or (100000,)
            return Text(shape=(None,), dtype="int64", max_shape=(100000,))
        dt = tf_dt.dtype.name
        if max_shape and len(max_shape) > len(tf_dt.shape):
            max_shape = max_shape[(len(max_shape) - len(tf_dt.shape)) :]

        max_shape = max_shape or tuple(
            10000 if dim is None else dim for dim in tf_dt.shape
        )
        return Tensor(shape=tf_dt.shape, dtype=dt, max_shape=max_shape)

    def image_to_hub(tf_dt, max_shape=None):
        dt = tf_dt.dtype.name
        if max_shape and len(max_shape) > len(tf_dt.shape):
            max_shape = max_shape[(len(max_shape) - len(tf_dt.shape)) :]

        max_shape = max_shape or tuple(
            10000 if dim is None else dim for dim in tf_dt.shape
        )
        return Image(
            shape=tf_dt.shape,
            dtype=dt,
            max_shape=max_shape,  # compressor="png"
        )

    def class_label_to_hub(tf_dt, max_shape=None):
        if hasattr(tf_dt, "_num_classes"):
            return ClassLabel(
                num_classes=tf_dt.num_classes,
            )
        else:
            return ClassLabel(names=tf_dt.names)

    def text_to_hub(tf_dt, max_shape=None):
        max_shape = max_shape or (100000,)
        dt = "int64"
        return Text(shape=(None,), dtype=dt, max_shape=max_shape)

    def bbox_to_hub(tf_dt, max_shape=None):
        dt = tf_dt.dtype.name
        return BBox(dtype=dt)

    def sequence_to_hub(tf_dt, max_shape=None):
        return Sequence(dtype=to_hub(tf_dt._feature), shape=())

    def audio_to_hub(tf_dt, max_shape=None):
        if max_shape and len(max_shape) > len(tf_dt.shape):
            max_shape = max_shape[(len(max_shape) - len(tf_dt.shape)) :]

        max_shape = max_shape or tuple(
            100000 if dim is None else dim for dim in tf_dt.shape
        )
        dt = tf_dt.dtype.name
        return Audio(
            shape=tf_dt.shape,
            dtype=dt,
            max_shape=max_shape,
            file_format=tf_dt._file_format,
            sample_rate=tf_dt._sample_rate,
        )

    def video_to_hub(tf_dt, max_shape=None):
        if max_shape and len(max_shape) > len(tf_dt.shape):
            max_shape = max_shape[(len(max_shape) - len(tf_dt.shape)) :]

        max_shape = max_shape or tuple(
            10000 if dim is None else dim for dim in tf_dt.shape
        )
        dt = tf_dt.dtype.name
        return Video(shape=tf_dt.shape, dtype=dt, max_shape=max_shape)

    my_schema = generate_schema(ds_info)

    def transform_numpy(sample):
        d = {}
        for k, v in sample.items():
            k = k.replace("/", "_")
            d[k] = transform_numpy(v) if isinstance(v, dict) else v.numpy()
        return d

    @hub.transform(schema=my_schema, scheduler=scheduler, workers=workers)
    def my_transform(sample):
        return transform_numpy(sample)

    return my_transform(ds)


class TorchDataset:
    def __init__(
        self,
        ds,
        transform=None,
        inplace=True,
        output_type=dict,
        indexes=None,
        key_list=None,
        shuffle=False,
    ):
        self._ds = None
        self._url = ds.url
        self._token = ds.token
        self._transform = transform
        self.inplace = inplace
        self.output_type = output_type
        self._inited = False
        self.key_list = key_list
        self.key_list = self.key_list or list(ds.keys)
        self.key_list = [
            key if key.startswith("/") else "/" + key for key in self.key_list
        ]
        for key in self.key_list:
            if key not in ds.keys:
                raise KeyError(key)

        self.max_chunk = self.get_max_chunk(ds)
        self.last_index = None
        if isinstance(indexes, int):
            self.last_index = indexes
        elif len(indexes) > 0:
            self.last_index = indexes[-1]
        self.indexes = self.shuffle_indexes(indexes, shuffle)

    def shuffle_indexes(self, indexes, shuffle):
        if not shuffle or isinstance(indexes, int):
            return indexes
        chunk_indexes_map = defaultdict(list)
        chunk_size = self.max_chunk
        for index in indexes:
            chunk = index // chunk_size
            chunk_indexes_map[chunk].append(index)
        chunk_indexes = list(chunk_indexes_map.values())
        random.shuffle(chunk_indexes)
        new_indexes = []
        for item in chunk_indexes:
            new_indexes.extend(item)
        return new_indexes

    def get_max_chunk(self, ds):
        max_chunk = 1
        for key, value in ds._tensors.items():
            if key in self.key_list:
                max_chunk = max(
                    max_chunk, ((None in value.shape) and 1 or value.chunks[0])
                )
        return max_chunk

    def _do_transform(self, data):
        return self._transform(data) if self._transform else data

    def _init_ds(self):
        """
        For each process, dataset should be independently loaded
        """
        if self._ds is None:
            self._ds = Dataset(self._url, token=self._token, lock_cache=False)
        if not self._inited:
            self._inited = True
            self._samples_in_chunks = {
                key: (None in value.shape) and 1 or value.chunks[0]
                for key, value in self._ds._tensors.items()
            }
            self._active_chunks = {}
            self._active_chunks_range = {}

    def __len__(self):
        self._init_ds()
        return len(self.indexes) if isinstance(self.indexes, list) else 1

    def _get_active_item(self, key, index):
        active_range = self._active_chunks_range.get(key)
        samples_per_chunk = self._samples_in_chunks[key]
        if active_range is None or index not in active_range:
            active_range_start = index - index % samples_per_chunk
            active_range = range(
                active_range_start,
                min(active_range_start + samples_per_chunk, self.last_index + 1),
            )
            self._active_chunks_range[key] = active_range
            self._active_chunks[key] = self._ds._tensors[key][
                active_range.start : active_range.stop
            ]
        return self._active_chunks[key][index % samples_per_chunk]

    def __getitem__(self, ind):
        if isinstance(self.indexes, int):
            if ind != 0:
                raise OutOfBoundsError(f"Got index {ind} for dataset of length 1")
            index = self.indexes
        else:
            index = self.indexes[ind]
        self._init_ds()
        d = {}
        for key in self._ds._tensors.keys():
            if key not in self.key_list:
                continue
            split_key = key.split("/")
            cur = d
            for i in range(1, len(split_key) - 1):
                if split_key[i] not in cur.keys():
                    cur[split_key[i]] = {}
                cur = cur[split_key[i]]

            item = self._get_active_item(key, index)
            if not isinstance(item, bytes) and not isinstance(item, str):
                t = item
                if self.inplace:
                    if t.dtype == "uint16":
                        t = t.astype("int32")
                    elif t.dtype == "uint32" or t.dtype == "uint64":
                        t = t.astype("int64")
                    t = torch.tensor(t)
                cur[split_key[-1]] = t
        d = self._do_transform(d)
        if self.inplace & (self.output_type != dict) & (isinstance(d, dict)):
            d = self.output_type(d.values())
        return d

    def __iter__(self):
        self._init_ds()
        for i in range(len(self)):
            yield self[i]


def _from_supervisely(project, scheduler: str = "single", workers: int = 1):
    try:
        import supervisely_lib as sly
        from supervisely_lib.project import project as sly_image_project
        from supervisely_lib.project import video_project as sly_video_project
        from skvideo.io import FFmpegReader, vread
    except ModuleNotFoundError:
        raise ModuleNotInstalledException("supervisely")
    if isinstance(project, str):
        with open(project + "meta.json") as meta_file:
            project_meta_dict = json.load(meta_file)
        instantiated = False
    else:
        project_meta_dict = project.meta.to_json()
        instantiated = True
    project_type = project_meta_dict["projectType"]
    mode = sly.OpenMode.READ

    def infer_image(paths):
        bboxes, masks = [], []
        classes_bb, classes_mask = [], []
        item_path, item_ann_path = paths

        ann = sly.Annotation.load_json_file(item_ann_path, project.meta)
        ann_dict = ann.to_json()
        sizes = (ann_dict["size"]["height"], ann_dict["size"]["width"])
        for obj in ann_dict["objects"]:
            if obj["geometryType"] == "rectangle":
                bboxes.append(
                    [item for sublist in obj["points"]["exterior"] for item in sublist]
                )
                classes_bb.append(obj["classTitle"])
            elif obj["geometryType"] == "polygon":
                img = PIL.Image.new("L", (sizes[1], sizes[0]), 0)
                PIL.ImageDraw.Draw(img).polygon(
                    [tuple(obj) for obj in obj["points"]["exterior"]],
                    outline=1,
                    fill=1,
                )
                masks.append(np.array(img))
                classes_mask.append(obj["classTitle"])
        return sizes, bboxes, masks, classes_bb, classes_mask

    def infer_video(paths):
        item_path, item_ann_path = paths
        vreader = FFmpegReader(item_path)
        return (vreader.getShape(),)

    def infer_project(project, project_type, read_mode):
        if project_type == "images":
            if not instantiated:
                project = sly_image_project.Project(project, mode)
            max_shape = (0, 0)
            return (
                project,
                Image,
                infer_image,
                max_shape,
            )
        elif project_type == "videos":
            if not instantiated:
                project = sly_video_project.VideoProject(project, mode)
            max_shape = (0, 0, 0, 0)
            return (
                project,
                Video,
                infer_video,
                max_shape,
            )

    project, main_blob, infer_ds, max_shape = infer_project(project, project_type, mode)

    image_paths = []
    label_names = []
    max_num_bboxes = 0
    max_num_polys = 0
    masks = False
    datasets = project.datasets.items()
    uniform = True
    for ds in datasets:
        for i, item in enumerate(ds):
            path = ds.get_item_paths(item)
            image_paths.append(path)
            inf = infer_ds(path)
            if len(inf) > 1:
                if inf[3]:
                    label_names.extend(inf[3])
                    if len(inf[3]) > max_num_bboxes:
                        max_num_bboxes = len(inf[3])
                if inf[4]:
                    label_names.extend(inf[4])
                    if len(inf[4]) > max_num_polys:
                        max_num_polys = len(inf[4])
                if inf[2]:
                    masks = True
            shape = inf[0]
            max_shape = np.maximum(shape, max_shape)
            if uniform and max_shape.any() and (shape != max_shape).any():
                uniform = False
    label_names = list(np.unique(label_names))
    items = chain(*datasets)
    idatasets = iter(datasets)
    ds, i = next(idatasets), 0
    key = "shape" if uniform else "max_shape"
    if project_type == "images":
        read = sly.imaging.image.read
        blob_shape = {key: (*max_shape.tolist(), 3)}
    elif project_type == "videos":
        read = vread
        blob_shape = {key: max_shape.tolist()}
        if key == "max_shape":
            blob_shape["shape"] = (None, None, None, 3)

    schema = {
        project_type: main_blob(**blob_shape),
    }
    if max_num_bboxes:
        schema["bbox"] = BBox(shape=(None, 4), max_shape=(max_num_bboxes, 4))
    if label_names:
        schema["label"] = ClassLabel(
            shape=(None,),
            max_shape=(max(max_num_bboxes, max_num_polys),),
            names=label_names,
        )
    if masks:
        schema["mask"] = Mask(
            shape=(None, None, None), max_shape=(*max_shape.tolist(), 1)
        )

    @hub.transform(schema=schema, scheduler=scheduler, workers=workers)
    def transformation(item):
        nonlocal i, ds
        sample = {}
        if i >= len(ds):
            ds, i = next(idatasets), 0
        item_path, item_ann_path = ds.get_item_paths(item)
        i += 1
        _, bboxes, masks, classes_bbox, classes_mask = infer_ds(
            (item_path, item_ann_path)
        )
        sample[project_type] = read(item_path)
        if bboxes:
            sample["bbox"] = np.array(bboxes)
            sample["label"] = [label_names.index(i) for i in classes_bbox]
        if masks:
            sample["mask"] = np.expand_dims(masks[0], -1)
            sample["label"] = [label_names.index(i) for i in classes_mask]
        return sample

    return transformation(list(items))


def _to_supervisely(dataset, output):
    try:
        import supervisely_lib as sly
        from skvideo.io import vwrite
    except ModuleNotFoundError:
        raise ModuleNotInstalledException("supervisely")
    schema_dict = dataset.schema.dict_
    for key, schem in schema_dict.items():
        if isinstance(schem, Image):
            project_type = "images"
            extension = "jpeg"
            break
        elif isinstance(schem, Video):
            project_type = "videos"
            extension = "mp4"
            break
    else:
        raise Exception
    mode = sly.OpenMode.CREATE
    if project_type == "images":
        _project = sly.Project
    elif project_type == "videos":
        _project = sly.VideoProject
    else:
        raise Exception
    pr = _project(output, mode)
    meta = pr.meta
    meta._project_type = project_type
    # probably here we can create multiple datasets
    out_ds = pr.create_dataset(output)
    try:
        fn_key = "filename"
        dataset[fn_key]
    except KeyError:
        fn_key = None
        zeroes = len(str(len(dataset)))
    for idx, view in enumerate(dataset):
        obj = view[key].compute()
        if fn_key:
            fn = view[fn_key].compute()
        else:
            fn = f"{idx:0{zeroes}}"
        fn = "{}.{}".format(fn, extension)
        # strangely supervisely prevents from using this method on videos
        try:
            out_ds.add_item_np(fn, obj)
        except RuntimeError:
            # fix with in-memory file
            path = "{}/{}".format(out_ds.item_dir, fn)
            vwrite(path, obj)
            out_ds._item_to_ann[fn] = fn + ".json"
            out_ds.set_ann(fn, out_ds._get_empty_annotaion(path))
    pr.set_meta(meta)
    return pr
