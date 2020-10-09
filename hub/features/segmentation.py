from typing import Tuple
import numpy as np

from hub.features.features import Tensor
from hub.features.class_label import ClassLabel


class Segmentation(Tensor):
    """`FeatureConnector` for segmentation
    """
    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        dtype: str = None,
        num_classes: int = None,
        names: Tuple[str] = None,
        names_file: str = None,
        max_shape: Tuple[int, ...] = None,
        chunks=True
    ):
        """Constructs a Segmentation FeatureConnector.
        Also constructs ClassLabel FeatureConnector for Segmentation classes.
        Args:
        shape: tuple of ints or None: (height, width, 1)
        dtype: dtype of segmentation array: `uint16` or `uint8`
        num_classes: `int`, number of classes. All labels must be < num_classes.
        names: `list<str>`, string names for the integer classes. The
                order in which the names are provided is kept.
        names_file: `str`, path to a file with names for the integer
                    classes, one per line.
        """
        super(Segmentation, self).__init__(shape, dtype, max_shape=max_shape, chunks=chunks)
        self.class_labels = ClassLabel(num_classes=num_classes, names=names, names_file=names_file, chunks=chunks)

    def get_segmentation_classes(self):
        class_indices = np.unique(self)
        return [self.class_labels.int2str(value) for value in class_indices]

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
