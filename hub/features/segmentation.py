from typing import Tuple, Union, Iterable
import numpy as np

from hub.features.features import Tensor
from hub.features.polygon import Polygon
from hub.features.class_label import ClassLabel


class Segmentation(Tensor):
    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        dtype: str = None,
        num_classes: int = None,
        names: Tuple[str] = None,
        names_file: str = None,
    ):
        super(Segmentation, self).__init__(shape, dtype)
        self.class_labels = ClassLabel(num_classes=num_classes, names=names,
                                      names_file=names_file)
    
    def get_segmentation_classes(self):
        class_indices = np.unique(self)
        return [self.class_labels.int2str(value) for value in class_indices]
    
    def get_attribute_dict(self):
        """Return class attributes
        """
        return self.__dict__
