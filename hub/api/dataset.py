from hub.api.tensor import Tensor
from hub.util.slice import merge_slices
from typing import Union, Dict
import numpy as np


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
        self.slice = ds_slice
        # TODO: read metadata and initialize tensors
        self.tensors: Dict[str, Tensor] = {}

    def __len__(self):
        """Return the greatest length of tensors"""
        return max(map(len, self.tensors.values()), default=0)

    def __getitem__(self, item: Union[slice, str]):
        if isinstance(item, str):
            return self.tensors[item]  # TODO: throw a pretty error
        elif isinstance(item, slice):
            new_slice = merge_slices(self.slice, item)
            return Dataset(self.path, self.mode, new_slice)
        else:
            return None  # TODO: throw a pretty error

    def __setitem__(self, item: Union[slice, str], value):
        if isinstance(item, str):
            if isinstance(value, np.ndarray):
                # TODO: write data and create tensor
                # self.tensors[item] = Tensor(...)
                return self.tensors[item]
            else:
                return None  # TODO: throw a pretty error
        else:
            return None  # TODO: throw a pretty error

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def from_path(path: str):
        """Create a local hub dataset from unstructured data.

        Note:
            This copies the data locally in hub format.
            Be careful when using this with large datasets.

        Args:
            path (str): Path to the data to be converted

        Returns:
            A Dataset instance whose path points to the hub formatted
            copy of the data.
        """

        return None  # TODO: hub.auto
