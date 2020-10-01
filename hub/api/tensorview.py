class TensorView:
    def __init__(
        self,
        dataset=None,
        subpath=None,
        slice_=None,
    ):

        assert dataset is not None
        assert subpath is not None
        assert slice_ is not None

        self.dataset = dataset
        self.subpath = subpath
        self.slice_ = slice_

    # TODO Add support for incomplete paths

    def numpy(self):
        if self.slice_ is None:
            return self.dataset_tensors[self.subpath][:]
        return self.dataset._tensors[self.subpath][self.slice_]

    # TODO Add slicing logic to tensorview
    def __getitem__(self, slice_):
        raise NotImplementedError()

    def __setitem__(self, slice_):
        raise NotImplementedError()
