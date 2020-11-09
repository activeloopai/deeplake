from typing import Tuple

from hub.features.features import Tensor


class Polygon(Tensor):
    """`HubFeature` for polygon"""

    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        dtype="int32",
        max_shape: Tuple[int, ...] = None,
        chunks=None,
        compressor="lz4",
    ):
        """Constructs a Polygon HubFeature.
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
        if isinstance(chunks, int):
            chunks = (chunks,)
        self._check_shape(shape)
        super().__init__(
            shape,
            dtype=dtype,
            max_shape=max_shape,
            chunks=chunks,
            compressor=compressor,
        )

    def _check_shape(self, shape):
        if len(shape) != 2 or shape[-1] != 2:
            raise ValueError("Wrong polygon shape provided")

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
