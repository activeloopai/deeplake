from hub.util.slice import merge_slices


class Tensor:
    def __init__(self, id: str, tensor_slice: slice = slice(None)):
        """
        Initialize a new tensor.
        This operation should normally only be performed by Hub internals.

        Parameters
        ----------
        id: str
            The internal identifier for this tensor.
        tensor_slice: slice, optional
            The slice object on which to restrict the view of this tensor.
        """
        self.id = id
        self.slice = tensor_slice
        self.shape = (0,)  # TODO: read metadata, get shape

    def __len__(self):
        """ Return the length of the primary axis """
        return self.shape[0]

    def __getitem__(self, tensor_slice: slice):
        new_slice = merge_slices(self.slice, tensor_slice)
        return Tensor(id, new_slice)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
