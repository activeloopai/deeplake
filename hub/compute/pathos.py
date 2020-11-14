import hub
from hub.utils import batchify
from hub.compute.transform import Transform

try:
    from pathos.pools import ProcessPool, ThreadPool
except Exception:
    pass


class PathosTransform(Transform):
    def __init__(self, func, schema, ds):
        Transform.__init__(self, func, schema, ds)
        Pool = ProcessPool or ThreadPool
        self.map = ThreadPool(nodes=2).map

    def store(self, url, token=None):
        """
        mary chunks with compute
        """
        ds = hub.Dataset(
            url, mode="w", shape=(len(self._ds),), schema=self._schema, token=token, cache=False
        )

        # Chunkwise compute
        batch_size = ds.chunksize

        def batchify_remote(ds):
            return tuple(batchify(ds, batch_size))

        def batched_func(i_xs):
            i, xs = i_xs
            print(xs)
            xs = [self._func(x) for x in xs]
            self._transfer_batch(ds, i, xs)

        batched = batchify_remote(ds)
        results = self.map(batched_func, enumerate(batched))

        #uploading can be done per chunk per item
        results = list(results)
        return ds

    def _transfer_batch(self, ds, i, results):
        for j, result in enumerate(results):
            for key in result:
                ds[key, i * ds.chunksize + j] = result[key]