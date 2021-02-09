"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Tuple


from hub.schema.features import Tensor


class Mask(Tensor):
    """`HubSchema` for mask

    | Usage:
    ----------
    >>> mask_tensor = Mask(shape=(300, 300, 1))
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        max_shape: Tuple[int, ...] = None,
        chunks=None,
        compressor="lz4",
    ):
        """Constructs a Mask HubSchema.

        Parameters
        ----------
        shape: tuple of ints or None
            Shape in format (height, width, 1)
        dtype: str
            Dtype of mask array. Default: `uint8`
        max_shape : Tuple[int]
            Maximum shape of tensor shape if tensor is dynamic
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks

        """
        self._check_shape(shape)
        super().__init__(
            shape,
            "bool_",
            max_shape=max_shape,
            chunks=chunks,
            compressor=compressor,
        )

    def _check_shape(self, shape):
        """Check if provided shape  maches mask characteristics."""
        if len(shape) != 3 or shape[-1] != 1:
            raise ValueError(
                "Wrong mask shape provided, should be of the format (height, width, 1) where height and width are integers or None"
            )

    def __str__(self):
        out = super().__str__()
        out = "Mask" + out[6:]
        return out

    def __repr__(self):
        return self.__str__()
