from hub.util.slice import merge_slices


class Dataset:
    def __init__(self, path: str, mode: str = "a", ds_slice: slice = slice(None)):
        """Initialize a new or existing dataset.

        Args:
            path (str): The location of the dataset.
                Can be a local path, or a url to a cloud storage provider.
            mode (str, optional): Mode in which the dataset is opened.
                Supported modes include ("r", "w", "a") plus an optional "+" suffix.
                Defaults to "a".
            ds_slice (slice, optional): The slice object restricting the view
                of this dataset's tensors. Defaults to slice(None, None, None).
                Used internally for iteration.
        """
        self.path = path
        self.mode = mode
        self.tensors = {}
        self.slice = ds_slice

    def __len__(self):
        """Return the greatest length of tensors"""
        return max(map(len, self.tensors.values()), default=0)

    def __getitem__(self, ds_slice):
        new_slice = merge_slices(self.slice, ds_slice)
        return Dataset(self.path, self.mode, new_slice)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
