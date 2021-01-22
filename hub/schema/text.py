from typing import Tuple

import numpy as np

from hub.schema.features import Tensor


class Text(Tensor):
    """`HubSchema` for text"""

    def __init__(
        self,
        shape: Tuple[int, ...] = (None,),
        dtype="int64",
        max_shape: Tuple[int, ...] = None,
        chunks=None,
        compressor="lz4",
    ):
        """| Construct the connector.
        Returns integer representation of given string.

        Parameters
        ----------
        shape: tuple of ints or None
            The shape of the text
        dtype: str
            the dtype for storage.
        max_shape : Tuple[int]
            Maximum number of words in the text
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks
        """
        self._set_dtype(dtype)
        super().__init__(
            shape,
            dtype,
            max_shape=max_shape,
            chunks=chunks,
            compressor=compressor,
        )

    def _set_dtype(self, dtype):
        """Set the dtype."""
        dtype = str(np.dtype(dtype))
        self.dtype = dtype

    def __str__(self):
        out = super().__str__()
        out = "Text" + out[6:]
        return out

    def __repr__(self):
        return self.__str__()
