import hub


class Transformer:
    def __init__(self, func, dtype, ds):
        self._func = func
        self._dtype = dtype
        self._ds = ds

    def __iter__(self):
        for item in self._ds:
            yield self._func(item)

    def store(self, url, token=None):
        ds = hub.open(url, mode="w", shape=self._ds.shape, dtype=self._dtype, token=token)
        for i, item in enumerate(self._ds):
            result = self._func(item)
            result["image"] = result["image"].squeeze()
            for key in result:
                ds[key, i] = result[key]
            # self._transfer(self._func(item), ds[i])
        return ds

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
