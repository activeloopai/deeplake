"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import sys
from hub_v1 import Dataset
from hub_v1.api.datasetview import DatasetView
from hub_v1.utils import batchify
from hub_v1.compute import Transform
from typing import Iterable, Iterator
from hub_v1.exceptions import ModuleNotInstalledException
from hub_v1.api.sharded_datasetview import ShardedDatasetView
import hub_v1
from hub_v1.api.dataset_utils import get_value, str_to_int


def empty_remote(template, **kwargs):
    """
    remote template
    """

    def wrapper(func):
        def inner(**kwargs):
            return func

        return inner

    return wrapper


try:
    import ray

    remote = ray.remote
except Exception:
    remote = empty_remote


class RayTransform(Transform):
    def __init__(self, func, schema, ds, scheduler="ray", workers=1, **kwargs):
        super(RayTransform, self).__init__(
            func, schema, ds, scheduler="single", workers=workers, **kwargs
        )
        self.workers = workers
        if "ray" not in sys.modules:
            raise ModuleNotInstalledException("ray")

        if not ray.is_initialized():
            ray.init(local_mode=True)

    @remote
    def _func_argd(_func, index, _ds, schema, kwargs):
        """
        Remote wrapper for user defined function
        """

        if isinstance(_ds, (Dataset, DatasetView)) and isinstance(_ds.indexes, int):
            _ds.indexes = [_ds.indexes]

        item = _ds[index]
        if isinstance(item, DatasetView) or isinstance(item, Dataset):
            item = item.compute()

        item = _func(0, item)
        item = Transform._flatten_dict(item, schema=schema)

        return list(item.values())

    def store(
        self,
        url: str,
        token: dict = None,
        length: int = None,
        ds: Iterable = None,
        progressbar: bool = True,
        public: bool = True,
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
        public: bool, optional
            only applicable if using hub storage, ignored otherwise
            setting this to False allows only the user who created it to access the dataset and
            the dataset won't be visible in the visualizer to the public

        Returns
        ----------
        ds: hub_v1.Dataset
            uploaded dataset
        """
        _ds = ds or self.base_ds

        num_returns = len(self._flatten_dict(self.schema, schema=self.schema).keys())
        results = [
            self._func_argd.options(num_returns=num_returns).remote(
                self.call_func, el, _ds, schema=self.schema, kwargs=self.kwargs
            )
            for el in range(len(_ds))
        ]

        if num_returns == 1:
            results = [[r] for r in results]

        results = self._split_list_to_dicts(results)

        ds = self.upload(
            results, url=url, token=token, progressbar=progressbar, public=public
        )
        return ds

    @remote
    def upload_chunk(i_batch, key, ds):
        """
        Remote function to upload a chunk
        Returns the shape of dynamic tensor to upload all in once after upload is completed

        Parameters
        ----------
        i_batch: Tuple
            Tuple composed of (index, batch)
        key: str
            Key of the tensor
        ds:
            Dataset to set to upload
        Returns
        ----------
        (key, slice_, shape) to set the shape later

        """
        i, batch = i_batch
        if not isinstance(batch, dict) and isinstance(batch[0], ray.ObjectRef):
            batch = ray.get(batch)
            # FIXME an ugly hack to unwrap elements with a schema that has one tensor
            num_returns = len(
                Transform._flatten_dict(ds.schema.dict_, schema=ds.schema.dict_).keys()
            )
            if num_returns == 1:
                batch = [item for sublist in batch for item in sublist]

        shape = None
        length = len(batch)

        slice_ = slice(i * length, (i + 1) * length)
        if ds[key].is_dynamic:
            # Sometimes ds._tensor slice_ gets out of the shape value
            shape = ds._tensors[f"/{key}"].get_shape_from_value([slice_], batch)
        ds[key, slice_] = batch

        return (key, [slice_], shape)

    def upload(
        self,
        results,
        url: str,
        token: dict,
        progressbar: bool = True,
        public: bool = True,
    ):
        """Batchified upload of results.
        For each tensor batchify based on its chunk and upload.
        If tensor is dynamic then still upload element by element.

        Parameters
        ----------
        dataset: hub_v1.Dataset
            Dataset object that should be written to
        results:
            Output of transform function
        progressbar: bool
        public: bool, optional
            only applicable if using hub storage, ignored otherwise
            setting this to False allows only the user who created it to access the dataset and
            the dataset won't be visible in the visualizer to the public
        Returns
        ----------
        ds: hub_v1.Dataset
            Uploaded dataset
        """
        if len(list(results.values())) == 0:
            shape = (0,)
        else:
            shape = (len(list(results.values())[0]),)

        ds = Dataset(
            url,
            mode="w",
            shape=shape,
            schema=self.schema,
            token=token,
            cache=False,
            public=public,
        )

        tasks = []
        for key, value in results.items():

            length = ds[key].chunksize[0]
            value = get_value(value)
            value = str_to_int(value, ds.tokenizer)
            batched_values = batchify(value, length)
            chunk_id = list(range(len(batched_values)))
            index_batched_values = list(zip(chunk_id, batched_values))

            ds._tensors[f"/{key}"].disable_dynamicness()

            results = [
                self.upload_chunk.remote(el, key=key, ds=ds)
                for el in index_batched_values
            ]
            tasks.extend(results)

        results = ray.get(tasks)
        self.set_dynamic_shapes(results, ds)
        ds.flush()
        return ds

    def set_dynamic_shapes(self, results, ds):
        """
        Sets shapes for dynamic tensors after the dataset is uploaded

        Parameters
        ----------
        results: Tuple
            results from uploading each chunk which includes (key, slice, shape) tuple
        ds:
            Dataset to set the shapes to
        Returns
        ----------
        """

        shapes = {}
        for (key, slice_, value) in results:
            if not ds[key].is_dynamic:
                continue

            if key not in shapes:
                shapes[key] = []
            shapes[key].append((slice_, value))

        for key, value in shapes.items():
            ds._tensors[f"/{key}"].enable_dynamicness()

            for (slice_, shape) in shapes[key]:
                ds._tensors[f"/{key}"].set_dynamic_shape(slice_, shape)


class TransformShard:
    def __init__(self, ds, func, schema, kwargs):

        if isinstance(ds, Dataset) or isinstance(ds, DatasetView):
            ds.squeeze_dim = False

        self._ds = ds
        self._func = func
        self.schema = schema
        self.kwargs = kwargs
        self.token = None

    def __call__(self, ids):
        """
        For each shard, transform each sample and then store inside shared memory of ray
        """
        for index in ids:
            item = self._ds[index]
            if isinstance(item, DatasetView) or isinstance(item, Dataset):
                item = item.compute()

            items = self._func(0, item)
            if not isinstance(items, list):
                items = [items]

            for item in items:
                yield Transform._flatten_dict(item, schema=self.schema)


class RayGeneratorTransform(RayTransform):
    def store(
        self,
        url: str,
        token: dict = None,
        length: int = None,
        ds: Iterable = None,
        progressbar: bool = True,
        public: bool = True,
    ):
        """
        The function to apply the transformation for each element by sharding the dataset

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
        public: bool, optional
            only applicable if using hub storage, ignored otherwise
            setting this to False allows only the user who created it to access the dataset and
            the dataset won't be visible in the visualizer to the public
        Returns
        ----------
        ds: hub_v1.Dataset
            uploaded dataset
        """
        _ds = ds or self.base_ds

        results = ray.util.iter.from_range(len(_ds), num_shards=self.workers).transform(
            TransformShard(
                ds=_ds, func=self.call_func, schema=self.schema, kwargs=self.kwargs
            )
        )

        @remote
        def upload_shard(i, shard_results):
            """
            Create a seperate dataset per shard
            """
            shard_results = self._split_list_to_dicts(shard_results)

            if len(shard_results) == 0 or len(list(shard_results.values())[0]) == 0:
                return None

            ds = self.upload(
                shard_results,
                url=f"{url}_shard_{i}",
                token=token,
                progressbar=progressbar,
                public=public,
            )
            return ds

        work = [
            upload_shard.remote(i, shard) for i, shard in enumerate(results.shards())
        ]

        datasets = ray.get(work)
        datasets = [d for d in datasets if d]
        ds = self.merge_sharded_dataset(datasets, url, token=token)
        return ds

    def merge_sharded_dataset(
        self,
        datasets,
        url,
        token=None,
    ):
        """
        Creates a sharded dataset and then applies ray transform to upload it
        """
        sharded_ds = ShardedDatasetView(datasets)

        @hub_v1.transform(schema=self.schema, scheduler="ray")
        def transform_identity(sample):
            return sample

        ds = transform_identity(sharded_ds).store(
            url,
            token=token,
        )

        return ds
