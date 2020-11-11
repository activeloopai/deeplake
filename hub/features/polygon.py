from typing import Tuple

from hub.features.features import Tensor


class Polygon(Tensor):
    """`FeatureConnector` for polygon

    | Usage: 
    ----------

    >>> polygon_tensor = Polygon(shape=(10, 2))
    >>> polygon_tensor = Polygon(shape=(None, 2))
    """
    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        max_shape: Tuple[int, ...] = None,
        chunks=True
    ):
        """Constructs a Polygon FeatureConnector.

        Parameters
        ----------
        shape: tuple of ints or None
            Shape in format (None, 2)
        max_shape : Tuple[int]
            Maximum shape of tensor shape if tensor is dynamic
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks

        Raises
        ----------
        ValueError: If the shape is invalid
        """
        self._check_shape(shape)
        super(Polygon, self).__init__(shape, dtype='uint32', max_shape=max_shape, chunks=chunks)

    def _check_shape(self, shape):
        """Check if provided shape  maches polygon characteristics.
        """
        if len(shape) != 2 or shape[-1] != 2:
            raise ValueError("Wrong polygon shape provided")

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
