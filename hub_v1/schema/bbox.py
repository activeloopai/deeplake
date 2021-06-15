"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Tuple
from hub_v1.schema.features import Tensor


class BBox(Tensor):
    """| HubSchema` for a normalized bounding box.

    Output: Tensor of type `float32` and shape `[4,]` which contains the
    normalized coordinates of the bounding box `[ymin, xmin, ymax, xmax]`


    Example: This example uploads a dataset with a Bounding box schema and retrieves it.

    ----------
    >>> import hub_v1
    >>> from hub_v1 import Dataset, schema
    >>> from hub_v1.schema import BBox
    >>> from numpy import asarray

    >>> tag = "username/dataset"
    >>>
    >>> # Create dataset
    >>> ds = Dataset(
    >>>   tag,
    >>>   shape=(10,),
    >>>   schema={
    >>>      "bbox": schema.BBox(dtype="uint8"),
    >>>  },
    >>> )
    >>>
    >>> ds["bbox", 1] = np.array([1,2,3,4])
    >>> ds.flush()

    >>> # Load data
    >>> ds = Dataset(tag)
    >>>
    >>> print(ds["bbox"][1].compute())
    [1 2 3 4]

    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (4,),
        max_shape: Tuple[int, ...] = None,
        dtype="float64",
        chunks=None,
        compressor="lz4",
    ):
        """Construct the connector.

        Parameters
        ----------
        shape: tuple of ints or None
            The shape of bounding box.
            Will be (4,) if only one bounding box corresponding to each sample.
            If N bboxes corresponding to each sample, shape should be (N,)
            If the number of bboxes for each sample vary from 0 to M. The shape should be set to (None, 4) and max_shape should be set to (M, 4)
            Defaults to (4,).
        max_shape : Tuple[int], optional
            Maximum shape of BBox
        dtype : str
                dtype of bbox coordinates. Default: 'float32'
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks
        """
        self.check_shape(shape)
        super(BBox, self).__init__(
            shape=shape,
            max_shape=max_shape,
            dtype=dtype,
            chunks=chunks,
            compressor=compressor,
        )

    def __str__(self):
        out = super().__str__()
        out = "BBox" + out[6:]
        return out

    def __repr__(self):
        return self.__str__()

    def check_shape(self, shape):
        if len(shape) not in [1, 2] or shape[-1] != 4:
            raise ValueError(
                "Wrong BBox shape provided, should be of the format (4,) or (None, 4) or (N, 4)"
            )
