class Dataset:
    def __init__(
        url: str = None, token=None, num_samples: int = None, mode: str = None, dtype = None,
    ):
        pass

    def __getitem__(self, slice_):
        raise NotImplementedError()

    def __setitem__(self, slice_, value):
        raise NotImplementedError()

    def commit(self):
        raise NotImplementedError()

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    @property
    def num_samples(self):
        raise NotImplementedError()

    @property
    def dtype(self):
        raise NotImplementedError()

    @shape.setter
    def num_samples(self, num_samples_: int):
        raise NotImplementedError()


def open(url: str = None, token: None, num_samples: int = None, mode: str = None) -> Dataset:
    raise NotImplementedError()
