from hub import Dataset
from hub.api.datasetview import DatasetView
from hub.utils import batchify
from hub.compute import Transform
from typing import Iterable

try:
    import ray

    remote = ray.remote
except Exception:

    def remote(template, **kwargs):
        """
        remote template
        """

        def wrapper(func):
            def inner(**kwargs):
                return func

            return inner

        return wrapper


class RayTransform(Transform):
    def __init__(self, func, schema, ds, scheduler="ray", nodes=1, **kwargs):
        super(RayTransform, self).__init__(
            func, schema, ds, scheduler="single", nodes=nodes, **kwargs
        )
        if not ray.is_initialized():
            ray.init()

    @remote
    def _func_argd(_func, index, _ds, schema, **kwargs):
        """
        Remote wrapper for user defined function
        """
        if isinstance(_ds, Dataset) or isinstance(_ds, DatasetView) :
            _ds.squeeze_dim = False

        item = _ds[index]
        item = _func(item, **kwargs)
        # item = self._unwrap(item) <- this will not work, need some tricks with ray
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

        num_returns = len(self._flatten_dict(self.schema, schema=self.schema).keys())
        results = [
            self._func_argd.options(num_returns=num_returns).remote(
                self._func, el, _ds, schema=self.schema, **self.kwargs
            )
            for el in range(len(_ds))
        ]
        results = self._split_list_to_dicts(results)
        ds = self.upload(results, url=url, token=token, progressbar=progressbar)
        return ds

    @remote
    def upload_chunk(i_batch, key, ds):
        """
        Remote function to upload a chunk
        """
        i, batch = i_batch
        length = len(batch)

        if not isinstance(batch, dict):
            batch = ray.get(batch)

        # FIXME replace below 8 lines with ds[key, i * length : (i + 1) * length] = batch
        if not ds[key].is_dynamic:
            if len(batch) != 1:
                ds[key, i * length : (i + 1) * length] = batch
            else:
                ds[key, i * length] = batch[0]
        else:
            for k, el in enumerate(batch):
                ds[key, i * length + k] = el

    def upload(self, results, url: str, token: dict, progressbar: bool = True):
        """Batchified upload of results
        For each tensor batchify based on its chunk and upload
        If tensor is dynamic then still upload element by element

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
