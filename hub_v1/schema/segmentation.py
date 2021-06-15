"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Tuple
import numpy as np

from hub_v1.schema.features import Tensor
from hub_v1.schema.class_label import ClassLabel


class Segmentation(Tensor):
    """`HubSchema` for segmentation"""

    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        dtype: str = None,
        num_classes: int = None,
        names: Tuple[str] = None,
        names_file: str = None,
        max_shape: Tuple[int, ...] = None,
        chunks=None,
        compressor="lz4",
    ):
        """Constructs a Segmentation HubSchema.
        Also constructs ClassLabel HubSchema for Segmentation classes.

        Parameters
        ----------
        shape: tuple of ints or None
            Shape in format (height, width, 1)
        dtype: str
            dtype of segmentation array: `uint16` or `uint8`
        num_classes: int
            Number of classes. All labels must be < num_classes.
        names: `list<str>`
            string names for the integer classes. The order in which the names are provided is kept.
        names_file: str
            Path to a file with names for the integer classes, one per line.
        max_shape : tuple[int]
            Maximum shape of tensor shape if tensor is dynamic
        chunks : tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks
        """
        super().__init__(shape, dtype, max_shape=max_shape, chunks=chunks)
        self.class_labels = ClassLabel(
            num_classes=num_classes,
            names=names,
            names_file=names_file,
            chunks=chunks,
            compressor="lz4",
        )

    def get_segmentation_classes(self):
        """Get classes of the segmentation mask"""
        return self.class_labels.names

    def __str__(self):
        out = super().__str__()
        out = "Segmentation" + out[6:-1]
        out = (
            out + ", names=" + str(self.class_labels._names)
            if self.class_labels._names is not None
            else out
        )
        out = (
            out + ", num_classes=" + str(self.class_labels._num_classes)
            if self.class_labels._num_classes is not None
            else out
        )
        out += ")"
        return out

    def __repr__(self):
        return self.__str__()
