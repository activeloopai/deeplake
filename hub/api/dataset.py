from hub.util.slice import merge_slices


class Dataset:
    def __init__(self, path: str, mode: str = "a", ds_slice: slice = slice(None)):
        """
        Initialize a new or existing dataset.

        Parameters
        ----------
        path: str
            The location of the dataset. Can be a local path, or a url to a cloud storage provider.
            Currently supported storage providers include [TODO].
        mode: str, optional (default to "a")
            Mode in which the dataset is opened. Supported modes include ("r", "w", "a") plus an optional "+" suffix.
        ds_slice: slice, optional (default to slice(None, None, None))
            The slice object on which to restrict the view of this dataset's tensors.
            Used internally for iteration.
        """
        self.path = path
        self.mode = mode
        self.tensors = {}
        self.slice = slice(None)

    def __len__(self):
        """Return the greatest length of tensors in a dataset"""
        return max(map(len, self.tensors.values()), default=0)

    def __getitem__(self, ds_slice):
        new_slice = merge_slices(self.slice, ds_slice)
        return Dataset(self.path, self.mode, new_slice)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
