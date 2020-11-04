import hub

import ray

from hub.utils import batch


class Transform:
    def __init__(self, func, schema, ds):
        self._func = func
        self._schema = schema
        self._ds = ds

    def __iter__(self):
        for item in self._ds:
            yield self._func(item)

    def store(self, url, token=None):
        ds = hub.Dataset(
            url, mode="w", shape=self._ds.shape, schema=self._schema, token=token
        )

        # Fire ray computes
        results = [self._func(item) for item in self._ds]

        # results = ray.get(results)
        for i, result in enumerate(results):
            for key in result:
                ds[key, i] = result[key]
        return ds

    def store_chunkwise(self, url, token=None):
        """
        mary chunks with compute
        """
        ds = hub.Dataset(
            url, mode="w", shape=self._ds.shape, schema=self._schema, token=token
        )

        results = [self._func.remote(item) for item in self._ds]

        # Chunkwise compute
        batch_size = ds.chunksize

        @ray.remote(num_returns=int(len(ds) / batch_size))
        def batchify(results):
            return tuple(batch(results, batch_size))

        results_batched = batchify.remote(results)
        if isinstance(results_batched, list):
            results = [
                self._transfer_batch.remote(self, ds, i, result)
                for i, result in enumerate(results_batched)
            ]
        else:
            self._transfer_batch.remote(self, ds, 0, results_batched)

        ray.get(results_batched)
        return ds

    @ray.remote
    def _transfer_batch(self, ds, i, results):
        for j, result in enumerate(results[0]):
            print(result)
            for key in result:
                ds[key, i * ds.chunksize + j] = result[key]

    def _transfer(self, from_, to):
        assert isinstance(from_, dict)
        for key, value in from_.items():
            to[key] = from_[key]

    def __getitem__(self, slice_):
        raise NotImplementedError()

    def __len__(self):
        return self.shape[0]

    @property
    def shape(self):
        return self._ds.shape

    @property
    def schema(self):
        return self._schema


def transform(schema):
    def wrapper(func):
        def inner(ds):
            return Transform(func, schema, ds)

        return inner

    return wrapper
