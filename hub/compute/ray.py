import hub
from hub.utils import batch
from hub.compute import Transform

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

    @remote
    def _transfer_batch(self, ds, i, results):
        for j, result in enumerate(results[0]):
            for key in result:
                ds[key, i * ds.chunksize + j] = result[key]

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

        @remote(num_returns=int(len(ds) / batch_size))
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