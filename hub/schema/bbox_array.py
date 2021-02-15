"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Tuple

from hub.schema.features import Tensor


class BBoxArray(Tensor):
    """| HubSchema` for a normalized bounding box.

    Output: Tensor of type `float32` and shape `[4,]` which contains the
    normalized coordinates of the bounding box `[ymin, xmin, ymax, xmax]`
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (None, 4),
        dtype="float64",
        max_shape: Tuple[int, ...] = None,
        chunks=None,
        compressor="lz4",
    ):
        """Construct the connector.

        Parameters
        ----------
        shape: tuple of ints or None
            The shape of BBoxArray:
            (num_bboxes, 4) where num_bboxes can be None.
            Defaults to (None, 4).
        dtype : str
                dtype of bbox coordinates. Default: 'float32'
        max_shape : Tuple[int]
            Maximum shape of tensor if tensor is dynamic
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks
        """
        self._check_shape(shape)
        super(BBoxArray, self).__init__(
            shape=shape,
            dtype=dtype,
            max_shape=max_shape,
            chunks=chunks,
            compressor=compressor,
        )

    def _check_shape(self, shape):
        """Check if provided shape matches BBoxArray characteristics."""
        if len(shape) != 2 or shape[-1] != 4:
            raise ValueError(
                "Wrong BBoxArray shape provided, should be of the format (num_bboxes, 4) where num_bboxes is an integer or None"
            )

    def __str__(self):
        out = super().__str__()
        out = "BBoxArray" + out[6:]
        return out

    def __repr__(self):
        return self.__str__()
