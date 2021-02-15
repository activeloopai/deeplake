"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Tuple
from hub.schema import Tensor
import numpy as np


class ImageArray(Tensor):
    """`HubSchema` for ImageArray

    The connector accepts as input a 4 dimensional `uint8` array
    representing a list or sequence of Images.

    Returns
    ----------
    Tensor: shape [num_images, height, width, channels],
         where channels must be 1 or 3
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (None, None, None, 3),
        dtype: str = "uint8",
        # TODO Add back encoding_format (probably named compress) when support for png and jpg support will be added
        max_shape: Tuple[int, ...] = None,
        # ffmpeg_extra_args=(),
        chunks=None,
        compressor="lz4",
    ):
        """Initializes the connector.

        Parameters
        ----------

        shape: tuple of ints
            The shape of the ImageArray (num_images, height, width,
            channels).
        dtype: `uint16` or `uint8` (default)
        max_shape : Tuple[int]
            Maximum shape of tensor if tensor is dynamic
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks

        Raises
        ----------
        ValueError: If the shape, dtype or encoding formats are invalid
        """
        self._set_dtype(dtype)
        self._check_shape(shape)
        super(ImageArray, self).__init__(
            dtype=dtype,
            shape=shape,
            max_shape=max_shape,
            chunks=chunks,
            compressor=compressor,
        )

    def __str__(self):
        out = super().__str__()
        out = "ImageArray" + out[6:]
        return out

    def __repr__(self):
        return self.__str__()

    def _set_dtype(self, dtype):
        """Set the dtype."""
        dtype = str(np.dtype(dtype))
        if dtype not in ("uint8", "uint16"):
            raise ValueError(f"Not supported dtype for {self.__class__.__name__}")
        self.dtype = dtype

    def _check_shape(self, shape):
        """Check if provided shape matches BBoxArray characteristics."""
        if len(shape) != 4:
            raise ValueError(
                "Wrong ImageArray shape provided, should be of the format (num_images, height, width, channels), where num_images, height, width, channels can be integer or None"
            )
