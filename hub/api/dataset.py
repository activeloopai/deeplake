from typing import Tuple

from hub.features import featurify, FeatureConnector, FlatTensor
from hub.store.storage_tensor import StorageTensor


class Dataset:
    def __init__(
        self,
        url: str = None,
        token=None,
        num_samples: int = None,
        mode: str = None,
        dtype=None,
    ):
        assert dtype is not None
        assert num_samples is not None
        assert url is not None
        assert mode is not None

        self.url = url
        self.token = token
        self.mode = mode
        self.num_samples = num_samples
        self.dtype: FeatureConnector = featurify(dtype)
        self._flat_tensors: Tuple[FlatTensor] = tuple(self.dtype._flatten())
        self._tensors = dict(self._generate_storage_tensors())

    def _generate_storage_tensors(self):
        for t in self._flat_tensors:
            t: FlatTensor = t
            yield t.path, StorageTensor(
                f"{self.url}{t.path}",
                shape=(self.num_samples,) + t.shape,
                dtype=t.dtype,
            )

    def _slice_split(self, slice_):
        path = slice_[0]
        assert isinstance(path, str)
        slice_ = slice_[1:]
        path = path if path.startswith("/") else "/" + path
        return path, slice_

    def __getitem__(self, slice_):
        path, slice_ = self._slice_split(slice_)
        return self._tensors[path][slice_]

    def __setitem__(self, slice_, value):
        path, slice_ = self._slice_split(slice_)
        self._tensors[path][slice_] = value

    def commit(self):
        raise NotImplementedError()

    def __exit__(self, type, value, traceback):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


def open(
    url: str = None, token=None, num_samples: int = None, mode: str = None
) -> Dataset:
    raise NotImplementedError()
