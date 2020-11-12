# from typing import Tuple

from hub.features.features import Tensor


class BBox(Tensor):
    """`HubFeature` for a normalized bounding box.
    Output:
    bbox: Tensor of type `float32` and shape `[4,]` which contains the
          normalized coordinates of the bounding box `[ymin, xmin, ymax, xmax]`
    """

    def __init__(self, dtype="float64", chunks=None):
        """Construct the connector.

        Parameters
        ----------
        dtype : str
                dtype of bbox coordinates. Default: 'float32'
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks
        """
        super(BBox, self).__init__(shape=(4,), dtype=dtype, chunks=chunks)

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
