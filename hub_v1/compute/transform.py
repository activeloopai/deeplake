"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import zarr
import numpy as np
import math
from typing import Dict, Iterable
from hub_v1.api.dataset import Dataset
from tqdm import tqdm
from collections.abc import MutableMapping
from hub_v1.utils import batchify
from hub_v1.api.dataset_utils import (
    get_value,
    slice_split,
    str_to_int,
    slice_extract_info,
)
import collections.abc as abc
from hub_v1.api.datasetview import DatasetView
from pathos.pools import ProcessPool, ThreadPool
from hub_v1.schema.sequence import Sequence
from hub_v1.schema.features import featurify
import os
from hub_v1.defaults import OBJECT_CHUNK


def get_sample_size(schema, workers):
    """Given Schema, decides how many samples to take at once and returns it"""
    schema = featurify(schema)
    samples = 10000
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

        samples = min(samples, (16 * 1024 * 1024 * 8) // (prod(shp) * sz))
    samples = max(samples, 1)
    return samples * workers


class Transform:
    def __init__(
        self, func, schema, ds, scheduler: str = "single", workers: int = 1, **kwargs
    ):
        """| Transform applies a user defined function to each sample in single threaded manner.

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
        self.workers = workers

        if isinstance(self._ds, Transform):
            self.base_ds = self._ds.base_ds
            self._func = self._ds._func[:]
            self._func.append(func)
            self.kwargs = self._ds.kwargs[:]
            self.kwargs.append(kwargs)
        else:
            self.base_ds = ds
            self._func = [func]
            self.kwargs = [kwargs]

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

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, slice_):
        """| Get an item to be computed without iterating on the whole dataset.
        | Creates a dataset view, then a temporary dataset to apply the transform.
        Parameters:
        ----------
        slice_: slice
            Gets a slice or slices from dataset
        """
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]

        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)

        slice_list = slice_list or [slice(None, None, None)]

        num, ofs = slice_extract_info(slice_list[0], self.shape[0])
        ds_view = self._ds[slice_list[0]]

        path = os.path.expanduser("~/.activeloop/tmparray")
        new_ds = self.store(path, length=num, ds=ds_view, progressbar=False)

        index = 1 if len(slice_) > 1 else 0
        slice_[index] = (
            slice(None, None, None) if not isinstance(slice_list[0], int) else 0
        )  # Get all shape dimension since we already sliced
        return new_ds[slice_]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @classmethod
    def _flatten_dict(self, d: Dict, parent_key="", schema=None):
        """| Helper function to flatten dictionary of a recursive tensor

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

    def _split_list_to_dicts(self, xs):
        """| Helper function that transform list of dicts into dicts of lists

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

    def _pbar(self, show: bool = True):
        """
        Returns a progress bar, if empty then it function does nothing
        """

        def _empty_pbar(xs, **kwargs):
            return xs

        single_threaded = self.map == map
        return tqdm if show and single_threaded else _empty_pbar

    def create_dataset(
        self, url: str, length: int = None, token: dict = None, public: bool = True
    ):
        """Helper function to create a dataset"""
        shape = (length,)
        ds = Dataset(
            url,
            mode="w",
            shape=shape,
            schema=self.schema,
            token=token,
            fs=zarr.storage.MemoryStore() if "tmp" in url else None,
            cache=False,
            public=public,
        )
        return ds

    def upload(self, results, ds: Dataset, token: dict, progressbar: bool = True):
        """Batchified upload of results.
        For each tensor batchify based on its chunk and upload.
        If tensor is dynamic then still upload element by element.
        For dynamic tensors, it disable dynamicness and then enables it back.

        Parameters
        ----------
        dataset: hub_v1.Dataset
            Dataset object that should be written to
        results:
            Output of transform function
        progressbar: bool
        Returns
        ----------
        ds: hub_v1.Dataset
            Uploaded dataset
        """
        offset = ds.indexes[
            0
        ]  # here ds.indexes will always be a contiguous list as obtained after slicing
        for key, value in results.items():
            chunk = ds[key].chunksize[0]
            chunk = 1 if chunk == 0 else chunk
            value = get_value(value)
            value = str_to_int(value, ds.dataset.tokenizer)

            num_chunks = len(value) / (chunk * self.workers)
            num_chunks = (
                1 + int(num_chunks) if num_chunks != int(num_chunks) else num_chunks
            )
            length = int(num_chunks * chunk) if self.workers != 1 else len(value)
            batched_values = (
                batchify(value, length, length + ((chunk - (offset % chunk))) % chunk)
                if length != len(value)
                else batchify(value, length)
            )
            len_batches = [len(item) for item in batched_values]

            def upload_chunk(i_batch):
                i, batch = i_batch
                length = len(batch)
                cur_offset = 0
                for it in range(i):
                    cur_offset += len_batches[it]
                slice_ = slice(cur_offset, cur_offset + length)
                ds[key, slice_] = batch

            index_batched_values = list(
                zip(list(range(len(batched_values))), batched_values)
            )

            # Disable dynamic arrays
            ds.dataset._tensors[f"/{key}"].disable_dynamicness()
            list(map(upload_chunk, index_batched_values))

            # Enable and rewrite shapes
            if ds.dataset._tensors[f"/{key}"].is_dynamic:
                ds.dataset._tensors[f"/{key}"].enable_dynamicness()
                ds.dataset._tensors[f"/{key}"].set_shape(
                    [slice(offset, offset + len(value))], value
                )

        ds.flush()
        return ds

    def call_func(self, fn_index, item, as_list=False):
        """Calls all the functions one after the other

        Parameters
        ----------
        fn_index: int
            The index starting from which the functions need to be called
        item:
            The item on which functions need to be applied
        as_list: bool, optional
            If true then treats the item as a list.

        Returns
        ----------
        result:
            The final output obtained after all transforms
        """
        result = item
        if fn_index < len(self._func):
            if as_list:
                result = [self.call_func(fn_index, it) for it in result]
            else:
                result = self._func[fn_index](result, **self.kwargs[fn_index])
                result = self.call_func(fn_index + 1, result, isinstance(result, list))
        result = self._unwrap(result) if isinstance(result, list) else result
        return result

    def store_shard(self, ds_in: Iterable, ds_out: Dataset, offset: int, token=None):
        """
        Takes a shard of iteratable ds_in, compute and stores in DatasetView
        """

        def _func_argd(item):
            if isinstance(item, DatasetView) or isinstance(item, Dataset):
                item = item.numpy()
            result = self.call_func(
                0, item
            )  # If the iterable obtained from iterating ds_in is a list, it is not treated as list
            return result

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
        sample_per_shard: int = None,
        public: bool = True,
    ):
        """| The function to apply the transformation for each element in batchified manner

        Parameters
        ----------
        url: str
            path where the data is going to be stored
        token: str or dict, optional
            If url is referring to a place where authorization is required,
            token is the parameter to pass the credentials, it can be filepath or dict
        length: int
            in case shape is None, user can provide length
        ds: Iterable
        progressbar: bool
            Show progress bar
        sample_per_shard: int
            How to split the iterator not to overfill RAM
        public: bool, optional
            only applicable if using hub storage, ignored otherwise
            setting this to False allows only the user who created it to access the dataset and
            the dataset won't be visible in the visualizer to the public
        Returns
        ----------
        ds: hub_v1.Dataset
            uploaded dataset
        """

        ds_in = ds or self.base_ds

        # compute shard length
        if sample_per_shard is None:
            n_samples = get_sample_size(self.schema, self.workers)
        else:
            n_samples = sample_per_shard
        try:
            length = len(ds_in) if hasattr(ds_in, "__len__") else n_samples
        except Exception:
            length = length or n_samples

        if length < n_samples:
            n_samples = length

        ds_out = self.create_dataset(url, length=length, token=token, public=public)

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
            desc=f"Computing the transformation in chunks of size {n_samples}",
        ) as pbar:
            for ds_in_shard in batchify_generator(ds_in, n_samples):
                n_results = self.store_shard(ds_in_shard, ds_out, start, token=token)
                total += n_results
                pbar.update(len(ds_in_shard))
                start += n_results

        ds_out.resize_shape(total)
        ds_out.flush()
        return ds_out

    @property
    def shape(self):
        return self._ds.shape
