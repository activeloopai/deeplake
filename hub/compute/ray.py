import sys
from hub import Dataset
from hub.api.datasetview import DatasetView
from hub.utils import batchify
from hub.compute import Transform
from typing import Iterable, Iterator
from hub.exceptions import ModuleNotInstalledException
from hub.api.sharded_datasetview import ShardedDatasetView
import hub


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
        if "ray" not in sys.modules:
            raise ModuleNotInstalledException("ray")

        if not ray.is_initialized():
            ray.init(local_mode=True)

    @remote
    def _func_argd(_func, index, _ds, schema, kwargs):
        """
        Remote wrapper for user defined function
        """

        if isinstance(_ds, Dataset) or isinstance(_ds, DatasetView):
            _ds.squeeze_dim = False

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
        Returns
        ----------
        ds: hub.Dataset
            uploaded dataset
        """
        _ds = ds or self._ds
        if isinstance(_ds, Transform):
            _ds = _ds.store(
                "{}_{}".format(url, _ds._func.__name__),
                token=token,
                progressbar=progressbar,
            )

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

        ds = self.upload(results, url=url, token=token, progressbar=progressbar)
        return ds

    @remote
    def upload_chunk(i_batch, key, ds):
        """
        Remote function to upload a chunk
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

        length = len(batch)
        if length != 1:
            ds[key, i * length : (i + 1) * length] = batch
        else:
            ds[key, i * length] = batch[0]

    def upload(self, results, url: str, token: dict, progressbar: bool = True):
        """Batchified upload of results.
        For each tensor batchify based on its chunk and upload.
        If tensor is dynamic then still upload element by element.

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
        )

        tasks = []
        for key, value in results.items():

            length = ds[key].chunksize[0]
            batched_values = batchify(value, length)
            chunk_id = list(range(len(batched_values)))
            index_batched_values = list(zip(chunk_id, batched_values))

            results = [
                self.upload_chunk.remote(el, key=key, ds=ds)
                for el in index_batched_values
            ]
            tasks.extend(results)

        ray.get(tasks)
        ds.commit()
        return ds


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
        Returns
        ----------
        ds: hub.Dataset
            uploaded dataset
        """
        _ds = ds or self._ds
        if isinstance(_ds, Transform):
            _ds = _ds.store(
                "{}_{}".format(url, _ds._func.__name__),
                token=token,
                progressbar=progressbar,
            )

        results = ray.util.iter.from_range(len(_ds), num_shards=4).transform(
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

        @hub.transform(schema=self.schema, scheduler="ray")
        def transform_identity(sample):
            return sample

        ds = transform_identity(sharded_ds).store(
            url,
            token=token,
        )

        return ds
