# from typing import Tuple

from hub.schema.features import Tensor


class BBox(Tensor):
    """| HubSchema` for a normalized bounding box.

    Output: Tensor of type `float32` and shape `[4,]` which contains the
    normalized coordinates of the bounding box `[ymin, xmin, ymax, xmax]`
    """

    def __init__(self, dtype="float64", chunks=None, compressor="lz4"):
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
        super(BBox, self).__init__(
            shape=(4,), dtype=dtype, chunks=chunks, compressor=compressor
        )

    def __str__(self):
        out = super().__str__()
        out = "BBox" + out[6:]
        return out

    def __repr__(self):
        return self.__str__()
