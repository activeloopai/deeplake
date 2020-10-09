from typing import Tuple

from hub.features.features import Tensor


class Polygon(Tensor):
    """`FeatureConnector` for polygon
    """
    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        max_shape: Tuple[int, ...] = None,
        chunks=True
    ):
        """Constructs a Polygon FeatureConnector.
        Args:
        shape: tuple of ints or None, i.e (None, 2)

        Example:
        ```python
        polygon_tensor = Polygon(shape=(10, 2))
        polygon_tensor = Polygon(shape=(None, 2))
        ```

        Raises:
        ValueError: If the shape is invalid
        """
        self._check_shape(shape)
        super(Polygon, self).__init__(shape, dtype='uint32', max_shape=max_shape, chunks=chunks)

    def _check_shape(self, shape):
        if len(shape) != 2 or shape[-1] != 2:
            raise ValueError("Wrong polygon shape provided")

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
