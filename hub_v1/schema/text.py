"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Tuple

import numpy as np

from hub_v1.schema.features import Tensor


class Text(Tensor):
    """Schema for text would define the shape and structure for the dataset.

    Output: `Tensor` of type `uint8` and shape `[height, width, num_channels]`
    for BMP, JPEG, and PNG images

    Example: This example uploads an `image` to a Hub dataset `image_dataset` with `HubSchema` and retrieves it.

    For data with fixed `shape`
    ----------
    >>> import hub_v1
    >>> from hub_v1 import Dataset, schema
    >>> from hub_v1.schema import Text

    >>> tag = "username/dataset"
    >>>
    >>> # Create dataset
    >>> ds = Dataset(
    >>>     tag,
    >>>     shape=(5,),
    >>>     schema = {
    >>>         "text": Text(shape=(11,)),
    >>>    },
    >>> )
    >>>
    >>> ds["text",0] = "Hello There"
    >>>
    >>> ds.flush()
    >>>
    >>> # Load the data
    >>> ds = Dataset(tag)
    >>>
    >>> print(ds["text"][0].compute())
    Hello There


    For data with variable `shape`, it is recommended to use `max_shape`

    >>> ds = Dataset(
    >>>     tag,
    >>>     shape=(5,),
    >>>     schema = {
    >>>         "text": Text(max_shape=(10,)),
    >>>    },
    >>> )
    >>>
    >>> ds["text",0] = "Welcome"
    >>> ds["text",1] = "to"
    >>> ds["text",2] = "Hub"
    >>>
    >>> ds.flush()
    >>>
    >>> # Load data
    >>> ds = Dataset(tag)
    >>>
    >>> print(ds["text"][0].compute())
    >>> print(ds["text"][1].compute())
    >>> print(ds["text"][2].compute())
    Welcome
    to
    Hub
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (None,),
        dtype="uint8",
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
