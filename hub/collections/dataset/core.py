from collections import abc
from configparser import ConfigParser
import json
import os
from typing import Dict, Tuple
import dask
import fsspec
import numpy as np
import traceback

from hub.client.hub_control import HubControlClient
from hub.codec import Base as BaseCodec
from hub.codec import from_name as codec_from_name
from hub.collections.tensor.core import Tensor
from hub.collections.client_manager import get_client, HubCache
from hub.log import logger
from hub.exceptions import PermissionException

try:
    import torch
except ImportError:
    pass

def _flatten(l):
    """
        Helper function to flatten the list
    """
    return [item for sublist in l for item in sublist]



class Transform:
    def __init__(self):
        pass

    def __call__(self, input):
        return self.forward(input)

    def meta(self):
        """
        Provides the metadata for all tensors including shapes, dtypes, dtags and chunksize for each array in the form
        
        Returns
        -------
        returns 
            dict of tensor
            
        Examples
        -------
        >>> def meta()
        >>>     return {
        >>>         ...
        >>>         "tesnor_name":{
        >>>             "shape": (1,256,256), 
        >>>             "dtype": "uint8", 
        >>>             "chunksize": 100, 
        >>>             "dtag": "segmentation"
        >>>         }
        >>>         ...
        >>>     }
        """

        raise NotImplementedError()

    def forward(input):
        """
        Takes a an element of a list or sample from dataset and returns sample of the dataset
        
        Parameters
        -------
        input
            an element of list or dict of arrays  
            
        Returns
        -------
        dict
            dict of numpy arrays
        
        Examples
        -------
        >>> def forward(input):
        >>>    ds = {}
        >>>    ds["image"] = np.empty(1, object)
        >>>    ds["image"][0] = np.array(256, 256)
        >>>    return ds
        """
        raise NotImplementedError()


DatasetGenerator = Transform


def _numpy_to_tuple(arr: np.ndarray):
    """ Converts numpy array to tuple of numpy arrays 
    """
    return [np.array([t]) for t in arr]


def _numpy_saver(
    fs: fsspec.AbstractFileSystem, filepath: str, array: np.ndarray, codec: BaseCodec
):
    """ Saves a single numpy array into filepath given specific filesystem
    """
    with fs.open(filepath, "wb") as f:
        f.write(codec.encode(array))


def _numpy_saver_multi(
    fs: fsspec.AbstractFileSystem, filepath: str, arrays: np.ndarray, offset: int
):
    for i in range(len(arrays)):
        _numpy_saver(fs, f"{filepath}/{offset+i}.npy", arrays[i : i + 1])
    return len(arrays)


def _preprocess_meta_before_save(meta: dict):
    meta = dict(meta)
    meta["dtype"] = str(meta["dtype"])
    return meta


def _dask_shape(input_shape: Tuple[int]):
    """ Dask accept np.nan value in shape if the axis length is not known, our API uses -1 for that, this function converts -1 to np.nan
    """
    return (np.nan,) + input_shape[1:] if input_shape[0] == -1 else input_shape


def _dict_to_tuple(d: dict):
    """ Converts dict of lists into (flattened list of values, list of keys)
    """
    keys = sorted(d.keys())
    lens = {len(d[key]) for key in keys}
    assert len(lens) == 1
    cnt = next(iter(lens))
    return [d[key][i] for i in range(cnt) for key in keys], keys


def _tuple_to_dict(t: tuple, keys: tuple):
    """ Converts (flattened list of values, list of keys) into dict of lists
    """
    cnt = len(keys)
    assert len(t) % cnt == 0
    return {key: [t[i] for i in range(j, len(t), cnt)] for j, key in enumerate(keys)}


def _load_creds(creds):
    """ Loads credentials from "{creds}" cfg file if such exists
    if creds is dict, then dict will be returned, assuming all credential data is in dict
    """
    if creds is None:
        return None
    elif isinstance(creds, str) and os.path.exists(creds):
        parser = ConfigParser()
        parser.read(creds)
        return {section: dict(parser.items(section)) for section in parser.sections()}
    else:
        return creds


def _connect(tag):
    """ Connects to the backend and receive credentials
    """

    creds = HubControlClient().get_config()
    dataset = HubControlClient().get_dataset_path(tag)

    if dataset and "path" in dataset:
        path = dataset["path"]
    else:
        sub_tags = tag.split("/")
        real_tag = sub_tags[-1]
        if len(sub_tags) > 1 and sub_tags[0] != creds["_id"]:
            username = creds["bucket"].split("/")[-1]
            creds["bucket"] = creds["bucket"].replace(username, sub_tags[0])

        path = f"{creds['bucket']}/{real_tag}"
    return path, creds


def _load_fs_and_path(path, creds=None, session_creds=True):
    """ Given url(path) and creds returns filesystem required for accessing that file + url's filepath in that filesystem
    """
    if (
        path.startswith("./")
        or path.startswith("/")
        or path.startswith("../")
        or path.startswith("~/")
    ):
        return fsspec.filesystem("file"), os.path.expanduser(path.replace("fs://", ""))

    if session_creds and creds is None and not path.startswith("s3://"):
        path, creds = _connect(path)

    if path.startswith("s3://"):
        path = path[5:]
        if creds is not None and session_creds:

            return (
                fsspec.filesystem(
                    "s3",
                    key=creds["access_key"],
                    secret=creds["secret_key"],
                    token=creds["session_token"],
                    client_kwargs={
                        "endpoint_url": creds["endpoint"],
                        "region_name": creds["region"],
                    },
                ),
                path,
            )
        elif creds is not None:
            return (
                fsspec.filesystem(
                    "s3", key=creds.get("access_key"), secret=creds.get("secret_key"),
                ),
                path,
            )
        else:
            return fsspec.filesystem("s3"), path


class Dataset:
    def __init__(self, tensors: Dict[str, Tensor]):
        """ Creates dict given dict of tensors (name -> Tensor key value pairs)
        """
        self._tensors = tensors
        shape = None
        for name, tensor in tensors.items():
            if shape is None or tensor.ndim > len(shape):
                shape = tensor.shape
            self._len = tensor.count

    def __len__(self) -> int:
        """ len of dataset (len of tensors across axis 0, yes, they all should be = to each other) 
        Raises Exception if length is unknown
        """
        if self._len == -1:
            raise Exception(
                "Cannot return __len__ of dataset for which __len__ is not known, use .count property, it will return -1 instead of this Exception"
            )
        return self._len

    @property
    def count(self) -> int:
        """ len of dataset (len of tensors across axis 0, yes, they all should be = to each other) 
        Returns -1 if length is unknown
        """
        return self._len

    def __iter__(self):
        """ Iterates over axis 0 return dict of Tensors
        """
        for i in range(len(self)):
            yield {key: t._array[i] for key, t in self._tensors.items()}

    def keys(self):
        """ Returns names of tensors
        """
        yield from self._tensors.keys()

    def values(self):
        """ Returns tensors
        """
        yield from self._tensors.values()

    def items(self):
        """ Returns tensors
        """
        yield from self._tensors.items()

    def __getitem__(self, slices) -> "Dataset":
        """ Returns a slice of dataset
        slices can be
            1) List of strs (slicing horizontally)
            2) List of slices or ints (slicing vertically)
            3) Both (1) and (2) at the same time
            4) Single int, slice, str is also accepted
        """
        if isinstance(slices, tuple):
            if all([isinstance(s, str) for s in slices]):
                return Dataset({key: self._tensors[key] for key in slices})
            elif isinstance(slices[0], abc.Iterable) and all(
                [isinstance(s, str) for s in slices[0]]
            ):
                return Dataset({key: self._tensors[key] for key in slices[0]})[
                    slices[1:]
                ]
            else:
                assert all(
                    [isinstance(s, slice) or isinstance(s, int) for s in slices]
                ), "invalid indexing, either wrong order or wrong type"
                ndim = len(slices)
                if all(isinstance(s, int) for s in slices):
                    return {
                        name: tensor[slices]
                        for name, tensor in self._tensors.items()
                        if tensor.ndim >= ndim
                    }
                else:
                    return Dataset(
                        {
                            name: tensor[slices]
                            for name, tensor in self._tensors.items()
                            if tensor.ndim >= ndim
                        }
                    )

        elif isinstance(slices, str):
            return self._tensors[slices]
        elif isinstance(slices, slice):
            return Dataset({key: value[slices] for key, value in self._tensors.items()})
        elif isinstance(slices, int):
            return {key: value[slices] for key, value in self._tensors.items()}

    def cache(self) -> "Dataset":
        raise NotImplementedError()

    def _store_unknown_sized_ds(self, fs: fsspec.AbstractFileSystem, path: str) -> int:
        client = get_client()
        worker_count = sum(client.ncores().values())
        # worker_count = 1
        chunks = {key: t._delayed_objs for key, t in self._tensors.items()}
        chunk_count = [len(items) for _, items in chunks.items()]
        assert (
            len(set(chunk_count)) == 1
        ), "Number of chunks in each tensor should be the same to be able to store dataset"
        chunk_count = chunk_count[0]
        count = 0
        collected = {el: None for el in self._tensors.keys()}
        collected_offset = {el: 0 for el in collected}
        # max_chunksize = max(*[t.chunksize for t in self._tensors])
        for i in range(0, chunk_count, worker_count):
            batch_count = min(i + worker_count, chunk_count) - i
            lasttime = True if i + worker_count >= chunk_count else False
            tasks = {
                key: delayed_objs[i : i + batch_count]
                for key, delayed_objs in chunks.items()
            }
            # logger.info(tasks)
            tasks, keys = _dict_to_tuple(tasks)

            # dask.visualize(
            #     tasks, filename=f"./data/tasks/{i}", optimize_graph=True,
            # )
            persisted = client.persist(tasks)
            persisted = _tuple_to_dict(persisted, keys)
            # for j in range(batch_count):
            #     assert (
            #         len(
            #             {
            #                 # len(objs[j])
            #                 # client.submit()
            #                 dask.delayed(len)(objs[j]).compute()
            #                 for objs in persisted.values()
            #             }
            #         )
            #         == 1
            #     ), "All numpy arrays returned from call should have same len"
            lens = {
                key: [dask.delayed(len)(objs[j]) for j in range(batch_count)]
                for key, objs in persisted.items()
            }
            lens, keys = _dict_to_tuple(lens)
            lens = client.gather(client.compute(lens))
            lens = _tuple_to_dict(lens, keys)
            for key, objs in persisted.items():
                arr = _dask_concat(
                    [
                        dask.array.from_delayed(
                            obj,
                            dtype=self._tensors[key].dtype,
                            shape=(lens[key][i],) + tuple(self._tensors[key].shape[1:]),
                        )
                        for i, obj in enumerate(objs)
                    ]
                )
                if collected[key] is None:
                    collected[key] = arr
                else:
                    collected[key] = _dask_concat([collected[key], arr])
            # tasks = [obj for key, objs in persisted.items() for obj in objs]
            tasks = []

            for key in list(collected.keys()):
                c = collected[key]
                chunksize = self._tensors[key].chunksize
                codec = codec_from_name(self._tensors[key].dcompress)
                cnt = len(c) - len(c) % chunksize if not lasttime else len(c)
                for i in range(0, cnt, chunksize):
                    tasks += [
                        dask.delayed(_numpy_saver)(
                            fs,
                            f"{path}/{key}/{collected_offset[key] + i}.npy",
                            c[i : i + chunksize],
                            codec,
                        )
                    ]
                collected_offset[key] += cnt
                collected[key] = collected[key][cnt:]
            client.gather(client.compute(tasks))
        count = set(collected_offset.values())
        assert (
            len(count) == 1
        ), "All tensors should be the same size to be stored in the same dataset"
        return next(iter(count))

    def _store_known_sized_ds(self, fs: fsspec.AbstractFileSystem, path: str) -> int:
        client = get_client()
        worker_count = sum(client.ncores().values())
        # chunksize = min(*[t.chunksize for t in self._tensors.values()])
        chunksize = (
            min(*[t.chunksize for t in self._tensors.values()])
            if len(self._tensors) > 1
            else next(iter(self._tensors.values())).chunksize
        )
        cnt = len(self)
        collected = {el: None for el in self._tensors.keys()}
        collected_offset = {el: 0 for el in collected}
        step = worker_count * chunksize
        for i in range(0, cnt, step):
            batch_count = min(step, cnt - i)
            lasttime = True if i + step >= cnt else False
            persisted = client.persist(
                [self._tensors[key]._array[i : i + batch_count] for key in collected]
            )
            persisted = {key: persisted[j] for j, key in enumerate(collected)}
            tasks = []
            for el, arr in persisted.items():
                if collected[el] is None:
                    collected[el] = arr
                else:
                    collected[el] = _dask_concat([collected[el], arr])
                c = collected[el]
                chunksize_ = self._tensors[el].chunksize
                codec = codec_from_name(self._tensors[el].dcompress)
                if len(c) >= chunksize_ or lasttime:
                    jcnt = len(c) - len(c) % chunksize_ if not lasttime else len(c)
                    for j in range(0, jcnt, chunksize_):
                        tasks += [
                            dask.delayed(_numpy_saver)(
                                fs,
                                f"{path}/{el}/{collected_offset[el] + j}.npy",
                                collected[el][j : j + chunksize_],
                                codec,
                            )
                        ]
                    collected_offset[el] += jcnt
                    collected[el] = collected[el][jcnt:]
            client.gather(client.compute(tasks))
        count = set(collected_offset.values())
        assert (
            len(count) == 1
        ), "All tensors should be the same size to be stored in the same dataset"
        return next(iter(count))

    @property
    def meta(self) -> dict:
        """ Dict of meta's of each tensor
        meta of tensor contains all metadata for tensor storage
        """
        tensor_meta = {
            name: _preprocess_meta_before_save(t._meta)
            for name, t in self._tensors.items()
        }
        ds_meta = {"tensors": tensor_meta, "len": self.count}
        return ds_meta

    def delete(self, tag, creds=None, session_creds=True) -> bool:
        """ Deletes dataset given tag(filepath) and credentials (optional)
        """
        fs, path = _load_fs_and_path(tag, creds, session_creds=session_creds)
        fs: fsspec.AbstractFileSystem = fs
        if fs.exists(path):
            fs.delete(path, recursive=True)
            return True
        return False

    def store(self, tag, creds=None, session_creds=True) -> "Dataset":
        """ Stores dataset by tag(filepath) given credentials (can be omitted)
        """
        fs, path = _load_fs_and_path(tag, creds, session_creds=session_creds)
        fs: fsspec.AbstractFileSystem = fs

        if (
            fs.exists(path)
            and not fs.exists(f"{path}/meta.json")
            and not fs.exists(f"{path}/HUB_DATASET")
            and len(fs.ls(path, detail=False)) > 0
        ):
            raise Exception(f"This path {path} is not a dataset path, tag: {tag}")
        self.delete(tag, creds)
        fs.makedirs(path)

        with fs.open(f"{path}/HUB_DATASET", "w") as f:
            f.write("Hello World")

        tensor_paths = [f"{path}/{t}" for t in self._tensors]
        for tensor_path in tensor_paths:
            fs.makedir(tensor_path)
        tensor_meta = {
            name: _preprocess_meta_before_save(t._meta)
            for name, t in self._tensors.items()
        }
        count = self.count
        try:
            if count == -1:
                count = self._store_unknown_sized_ds(fs, path)
            else:
                self._store_known_sized_ds(fs, path)
        except PermissionError as e:
            logger.error(e)
            raise PermissionException(tag)

        for _, el in tensor_meta.items():
            el["shape"] = (count,) + tuple(el["shape"][1:])
        ds_meta = {"tensors": tensor_meta, "len": count}
        with fs.open(f"{path}/meta.json", "w") as f:
            f.write(json.dumps(ds_meta, indent=2, sort_keys=True))

        return load(tag, creds)

    def to_pytorch(self, transform=None):
        """
            Transforms into pytorch dataset
            
            Parameters
            ----------
            transform: func
                any transform that takes input a dictionary of a sample and returns transformed dictionary 
        """
        return TorchDataset(self, transform)

    def to_tensorflow(self):
        """
            Transforms into tensorflow dataset
        """
        try:
            import tensorflow as tf
        except ImportError:
            pass

        def tf_gen(step=4):
            with dask.config.set(scheduler="sync"):
                for index in range(0, len(self), step):
                    arrs = [self[index : index + step].values() for i in range(1)]
                    arrs = list(map(lambda x: x._array, _flatten(arrs)))
                    arrs = dask.delayed(list, pure=False, nout=len(list(self.keys())))(
                        arrs
                    )
                    arrs = arrs.compute()
                    for i in range(step):
                        sample = {key: r[i] for key, r in zip(self[index].keys(), arrs)}
                        yield sample

        def tf_dtype(np_dtype):
            try:
                return tf.dtypes.as_dtype(np_dtype)
            except Exception as e:
                return tf.variant

        # TODO use None for dimensions you don't know the length tf.TensorShape([None])
        # FIXME Dataset Generator is not very good with multiprocessing but its good for fast tensorflow support
        return tf.data.Dataset.from_generator(
            tf_gen,
            output_types={
                key: tf_dtype(self._tensors[key].dtype) for key in self.keys()
            },
            output_shapes={key: self._tensors[key].shape[1:] for key in self.keys()},
        )


def _numpy_load(
    fs: fsspec.AbstractFileSystem, filepath: str, codec: BaseCodec
) -> np.ndarray:
    """ Given filesystem and filepath, loads numpy array
    """
    # assert fs.exists(
    #    filepath
    # ), f"Dataset file {filepath} does not exists. Your dataset data is likely to be corrupted"

    try:
        with fs.open(filepath, "rb") as f:
            return codec.decode(f.read())
    except Exception as e:
        logger.error(traceback.format_exc() + str(e))
        raise Exception(
            f"Dataset file {filepath} does not exists. Your dataset data is likely to be corrupted"
        )


def load(tag, creds=None, session_creds=True) -> Dataset:
    """ Load a dataset from repository using given url and credentials (optional)
    """
    fs, path = _load_fs_and_path(tag, creds, session_creds=session_creds)
    fs: fsspec.AbstractFileSystem = fs
    path_2 = f"{path}/meta.json"
    if not fs.exists(path):
        from hub.exceptions import DatasetNotFound

        raise DatasetNotFound(tag)

    with fs.open(path_2, "r") as f:
        ds_meta = json.loads(f.read())

    for name in ds_meta["tensors"]:
        assert fs.exists(
            f"{path}/{name}"
        ), f"Tensor {name} of {tag} dataset does not exist"
    if ds_meta["len"] == 0:
        logger.warning("The dataset is empty (has 0 samples)")
        return Dataset(
            {
                name: Tensor(
                    tmeta,
                    dask.array.from_array(
                        np.empty(shape=(0,) + tuple(tmeta["shape"][1:]), dtype="uint8"),
                    ),
                )
                for name, tmeta in ds_meta["tensors"].items()
            }
        )
    len_ = ds_meta["len"]

    # added reverse compatibility for previous versions
    for name, tmeta in ds_meta["tensors"].items():
        if "chunksize" not in tmeta:
            tmeta["chunksize"] = 1

    return Dataset(
        {
            name: Tensor(
                tmeta,
                _dask_concat(
                    [
                        dask.array.from_delayed(
                            dask.delayed(_numpy_load)(
                                fs,
                                f"{path}/{name}/{i}.npy",
                                codec_from_name(tmeta.get("dcompress")),
                            ),
                            shape=(min(tmeta["chunksize"], len_ - i),)
                            + tuple(tmeta["shape"][1:]),
                            dtype=tmeta["dtype"],
                        )
                        for i in range(0, len_, tmeta["chunksize"])
                    ]
                ),
            )
            for name, tmeta in ds_meta["tensors"].items()
        }
    )


def _is_arraylike(arr):
    return (
        isinstance(arr, np.ndarray) or isinstance(arr, list) or isinstance(arr, tuple)
    )


def _is_tensor_dynamic(tensor):
    arr = tensor._array.to_delayed().flatten()[0].compute()
    return str(tensor.dtype) == "object" and _is_arraylike(arr.flatten()[0])


class TorchDataset:
    def __init__(self, ds, transform=None):
        self._ds = ds
        self._transform = transform
        self._dynkeys = {
            key for key in self._ds.keys() if _is_tensor_dynamic(self._ds[key])
        }

        def cost(nbytes, time):
            print(nbytes, time)
            return float(time) / (nbytes or 1) / 1e9

        self.client = None

    def _do_transform(self, data):
        return self._transform(data) if self._transform else data

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index):
        with dask.config.set(scheduler="sync", delayed_pure=True):
            arrs = [self._ds[index : index + 1].values() for i in range(1)]
            arrs = list(map(lambda x: x._array, _flatten(arrs)))
            arrs = dask.delayed(list, pure=True, nout=len(list(self._ds.keys())))(arrs)
            arrs = arrs.compute()

            arrs = {key: r[0] for key, r in zip(self._ds[index].keys(), arrs)}

        objs = self._do_transform(arrs)
        if isinstance(objs, dict):
            objs = {k: self._to_tensor(k, v) for k, v in objs.items()}
        elif isinstance(objs, list):
            objs = [self._to_tensor(v) for v in objs]
        return objs

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _to_tensor(self, key, sample):
        if key not in self._dynkeys:
            return torch.tensor(sample)
        else:
            return [torch.tensor(item) for item in sample]

    def collate_fn(self, batch):
        batch = tuple(batch)
        keys = tuple(batch[0].keys())
        ans = {key: [item[key] for item in batch] for key in keys}

        for key in keys:
            if key not in self._dynkeys:
                ans[key] = torch.stack(ans[key], dim=0, out=None)

        return ans


def _dask_concat(arr):
    if len(arr) == 1:
        return arr[0]
    else:
        return dask.array.concatenate(arr)


# class TensorflowDataset(tfds.core.GeneratorBasedBuilder):
#     def _info(self):
#         return tfds.core.DatasetInfo(
#             builder=self,
#             # This is the description that will appear on the datasets page.
#             description=(
#                 "This is the dataset for xxx. It contains yyy. The "
#                 "images are kept at their original dimensions."
#             ),
#             # tfds.features.FeatureConnectors
#             # features=tfds.features.FeaturesDict(
#             #     {
#             #         "image_description": tfds.features.Text(),
#             #         "image": tfds.features.Image(),
#             #         # Here, labels can be of 5 distinct values.
#             #         "label": tfds.features.ClassLabel(num_classes=5),
#             #     }
#             # ),
#             # If there's a common (input, target) tuple from the features,
#             # specify them here. They'll be used if as_supervised=True in
#             # builder.as_dataset.
#             # supervised_keys=("image", "label"),
#             # Homepage of the dataset for documentation
#             homepage="https://dataset-homepage.org",
#             # Bibtex citation for the dataset
#             citation=r"""@article{my-awesome-dataset-2020,
#                                 author = {Smith, John},"}""",
#         )
