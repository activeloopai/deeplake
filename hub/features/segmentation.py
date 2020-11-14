from typing import Tuple
import numpy as np

from hub.features.features import Tensor
from hub.features.class_label import ClassLabel


class Segmentation(Tensor):
    """`HubFeature` for segmentation"""

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
        """Constructs a Segmentation HubFeature.
        Also constructs ClassLabel HubFeature for Segmentation classes.

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
        class_indices = np.unique(self)
        return [self.class_labels.int2str(value) for value in class_indices]

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
