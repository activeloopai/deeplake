import zarr
import numpy as np
import math
from psutil import virtual_memory
from typing import Dict, Iterable
from hub.api.dataset import Dataset
from tqdm import tqdm
from collections.abc import MutableMapping
from hub.utils import batchify
from hub.api.dataset_utils import slice_extract_info, slice_split, str_to_int
import collections.abc as abc
from hub.api.datasetview import DatasetView
from pathos.pools import ProcessPool, ThreadPool
from hub.schema import Primitive
from hub.schema.sequence import Sequence
from hub.schema.features import featurify
import posixpath
from hub.defaults import OBJECT_CHUNK

from hub.utils import Timer


def get_sample_size_in_memory(schema):
    """Given Schema, looks into memory how many samples can fit and returns it"""
    schema = featurify(schema)
    mem = virtual_memory()
    sample_size = 0
    for feature in schema._flatten():
        shp = list(feature.max_shape)
        if len(shp) == 0:
            shp = [1]

        sz = np.dtype(feature.dtype).itemsize
        if feature.dtype == "object":
            sz = (16 * 1024 * 1024 * 8) / 128

        def prod(shp):
            res = 1
            for s in shp:
                res *= s
            return res

        sample_size += prod(shp) * sz

    if sample_size > mem.total:
        return 1

    return int(mem.total // sample_size)


class Transform:
    def __init__(
        self, func, schema, ds, scheduler: str = "single", workers: int = 1, **kwargs
    ):
        """
        Transform applies a user defined function to each sample in single threaded manner

        Parameters
        ----------
        func: function
            user defined function func(x, **kwargs)
        schema: dict of dtypes
            the structure of the final dataset that will be created
        ds: Iterative
            input dataset or a list that can be iterated
        scheduler: str
            choice between "single", "threaded", "processed"
        workers: int
            how many threads or processes to use
        **kwargs:
            additional arguments that will be passed to func as static argument for all samples
        """
        self._func = func
        self.schema = schema
        self._ds = ds
        self.kwargs = kwargs

        if scheduler == "threaded" or (scheduler == "single" and workers > 1):
            self.map = ThreadPool(nodes=workers).map
        elif scheduler == "processed":
            self.map = ProcessPool(nodes=workers).map
        elif scheduler == "single":
            self.map = map
        elif scheduler == "ray":
            try:
                from ray.util.multiprocessing import Pool as RayPool
            except Exception:
                pass
            self.map = RayPool().map
        else:
            raise Exception(
                f"Scheduler {scheduler} not understood, please use 'single', 'threaded', 'processed'"
            )

    @classmethod
    def _flatten_dict(self, d: Dict, parent_key="", schema=None):
        """
        Helper function to flatten dictionary of a recursive tensor

        Parameters
        ----------
        d: dict
        """
        items = []
        for k, v in d.items():
            new_key = parent_key + "/" + k if parent_key else k
            if isinstance(v, MutableMapping) and not isinstance(
                self.dtype_from_path(new_key, schema), Sequence
            ):
                items.extend(
                    self._flatten_dict(v, parent_key=new_key, schema=schema).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)

    @classmethod
    def _flatten(cls, items, schema):
        """
        Takes a dictionary or list of dictionary
        Returns a dictionary of concatenated values
        Dictionary follows schema
        """
        final_item = {}
        for item in cls._unwrap(items):
            item = cls._flatten_dict(item, schema=schema)

            for k, v in item.items():
                if k in final_item:
                    final_item[k].append(v)
                else:
                    final_item[k] = [v]
        return final_item

    @classmethod
    def dtype_from_path(cls, path, schema):
        """
        Helper function to get the dtype from the path
        """
        path = path.split("/")
        cur_type = schema
        for subpath in path[:-1]:
            cur_type = cur_type[subpath]
            cur_type = cur_type.dict_
        return cur_type[path[-1]]

    def _split_list_to_dicts(self, xs):
        """
        Helper function that transform list of dicts into dicts of lists

        Parameters
        ----------
        xs: list of dicts

        Returns
        ----------
        xs_new: dicts of lists
        """
        xs_new = {}
        for x in xs:
            if isinstance(x, list):
                x = dict(
                    zip(self._flatten_dict(self.schema, schema=self.schema).keys(), x)
                )

            for key, value in x.items():
                if key in xs_new:
                    xs_new[key].append(value)
                else:
                    xs_new[key] = [value]
        return xs_new

    def create_dataset(self, url, length=None, token=None):
        """Helper function to creat a dataset"""
        shape = (length,)
        ds = Dataset(
            url,
            mode="w",
            shape=shape,
            schema=self.schema,
            token=token,
            fs=zarr.storage.MemoryStore() if "tmp" in url else None,
            cache=False,
        )
        return ds

    def upload(self, results, ds: Dataset, token: dict, progressbar: bool = True):
        """Batchified upload of results
        For each tensor batchify based on its chunk and upload
        If tensor is dynamic then still upload element by element
        For dynamic tensors, it disable dynamicness and then enables it back

        Parameters
        ----------
        dataset: hub.Dataset
            Dataset object that should be written to
        results:
            Output of transform function
        progressbar: bool
        Returns
        ----------
        ds: hub.Dataset
            Uploaded dataset
        """

        for key, value in results.items():

            length = ds[key].chunksize[0]
            value = str_to_int(value, ds.dataset.tokenizer)

            if length == 0:
                length = 1

            batched_values = batchify(value, length)

            def upload_chunk(i_batch):
                i, batch = i_batch
                batch_length = len(batch)
                if batch_length != 1:
                    ds[key, i * length : i * length + batch_length] = batch
                else:
                    ds[key, i * length] = batch[0]

            index_batched_values = list(
                zip(list(range(len(batched_values))), batched_values)
            )

            # Disable dynamic arrays
            ds.dataset._tensors[f"/{key}"].disable_dynamicness()
            list(self.map(upload_chunk, index_batched_values))

            # Enable and rewrite shapes
            if ds.dataset._tensors[f"/{key}"].is_dynamic:
                ds.dataset._tensors[f"/{key}"].enable_dynamicness()
                [
                    ds.dataset._tensors[f"/{key}"].set_shape([i + ds.offset], v)
                    for i, v in enumerate(value)
                ]

        ds.commit()
        return ds

    def _pbar(self, show: bool = True):
        """
        Returns a progress bar, if empty then it function does nothing
        """

        def _empty_pbar(xs, **kwargs):
            return xs

        single_threaded = self.map == map
        return tqdm if show and single_threaded else _empty_pbar

    @classmethod
    def _unwrap(cls, results):
        """
        If there is any list then unwrap it into its elements
        """
        items = []
        for r in results:
            if isinstance(r, dict):
                items.append(r)
            else:
                items.extend(r)
        return items

    def store_shard(self, ds_in: Iterable, ds_out: Dataset, offset: int, token=None):
        """
        Takes a shard of iteratable ds_in, compute and stores in DatasetView
        """

        def _func_argd(item):
            return self._func(item, **self.kwargs)

        ds_in = list(ds_in)
        results = self.map(
            _func_argd,
            ds_in,
        )
        results = self._unwrap(results)
        results = self.map(lambda x: self._flatten_dict(x, schema=self.schema), results)
        results = list(results)

        results = self._split_list_to_dicts(results)

        results_values = list(results.values())
        if len(results_values) == 0:
            return 0

        n_results = len(results_values[0])
        if n_results == 0:
            return 0

        additional = max(offset + n_results - ds_out.shape[0], 0)

        ds_out.append_shape(additional)

        self.upload(
            results,
            ds_out[offset : offset + n_results],
            token=token,
        )

        return n_results

    def store(
        self,
        url: str,
        token: dict = None,
        length: int = None,
        ds: Iterable = None,
        progressbar: bool = True,
        sample_per_shard=None,
    ):
        """
        The function to apply the transformation for each element in batchified manner

        Parameters
        ----------
        url: str
            path where the data is going to be stored
        token: str or dict, optional
            If url is refering to a place where authorization is required,
            token is the parameter to pass the credentials, it can be filepath or dict
        length: int
            in case shape is None, user can provide length
        ds: Iterable
        progressbar: bool
            Show progress bar
        sample_per_shard: int
            How to split the iterator not to overfill RAM
        Returns
        ----------
        ds: hub.Dataset
            uploaded dataset
        """

        ds_in = ds or self._ds
        if isinstance(ds_in, Transform):
            ds_in = ds_in.store(
                "{}_{}".format(url, ds_in._func.__name__),
                token=token,
                progressbar=progressbar,
            )

        # compute shard length
        if sample_per_shard is None:
            n_samples = get_sample_size_in_memory(self.schema)
            n_samples = min(10000, n_samples)
            n_samples = max(512, n_samples)
        else:
            n_samples = sample_per_shard

        try:
            length = len(ds_in) if hasattr(ds_in, "__len__") else n_samples
        except Exception:
            length = n_samples

        if length < n_samples:
            n_samples = length

        ds_out = self.create_dataset(url, length=length, token=token)

        def batchify_generator(iterator: Iterable, size: int):
            batch = []
            for el in iterator:
                batch.append(el)
                if len(batch) >= size:
                    yield batch
                    batch = []
            yield batch

        start = 0
        total = 0

        with tqdm(
            total=length,
            unit_scale=True,
            unit=" items",
            desc="Computing the transormation",
        ) as pbar:
            pbar.update(length // 10)
            for ds_in_shard in batchify_generator(ds_in, n_samples):

                n_results = self.store_shard(ds_in_shard, ds_out, start, token=token)
                total += n_results

                if n_results < n_samples or n_results == 0:
                    break
                start += n_samples
                pbar.update(n_samples)

        ds_out.resize_shape(total)
        ds_out.commit()
        return ds_out

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, slice_):
        """
        Get an item to be computed without iterating on the whole dataset
        Creates a dataset view, then a temporary dataset to apply the transform

        slice_: slice
            Gets a slice or slices from dataset
        """
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]

        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)

        if len(slice_list) == 0:
            slice_list = [slice(None, None, None)]

        num, ofs = slice_extract_info(slice_list[0], self.shape[0])

        ds_view = DatasetView(
            dataset=self._ds,
            num_samples=num,
            offset=ofs,
            squeeze_dim=isinstance(slice_list[0], int),
        )

        path = posixpath.expanduser("~/.activeloop/tmparray")
        new_ds = self.store(path, length=num, ds=ds_view, progressbar=False)

        index = 1 if len(slice_) > 1 else 0
        slice_[index] = (
            slice(None, None, None) if not isinstance(slice_list[0], int) else 0
        )  # Get all shape dimension since we already sliced
        return new_ds[slice_]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @property
    def shape(self):
        return self._ds.shape
