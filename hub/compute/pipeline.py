import hub

import ray


class Transformer:
    def __init__(self, func, dtype, ds):
        self._func = ray.remote(func)
        self._dtype = dtype
        self._ds = ds

    def __iter__(self):
        for item in self._ds:
            yield self._func.remote(item)

    def store(self, url, token=None):
        ds = hub.open(
            url, mode="w", shape=self._ds.shape, dtype=self._dtype, token=token
        )
        # TODO chunk based compute combination and storage
        results = [self._func.remote(item) for item in self._ds]
        results = ray.get(results)
        results = [self._transfer_2(ds, i, result) for i, result in enumerate(results)]
        # ray.get(results)
        # self._transfer(self._func(item), ds[i])
        return ds

    # @ray.remote
    def _transfer_2(self, ds, i, result):
        for key in result:
            ds[key, i] = result[key]

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
    def dtype(self):
        return self._dtype


def transform(dtype):
    def wrapper(func):
        def inner(ds):
            return Transformer(func, dtype, ds)

        return inner

    return wrapper
