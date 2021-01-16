from typing import Tuple

from hub.schema.features import Tensor


class Polygon(Tensor):
    """`HubSchema` for polygon

    | Usage:
    ----------

    >>> polygon_tensor = Polygon(shape=(10, 2))
    >>> polygon_tensor = Polygon(shape=(None, 2))
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        dtype="int32",
        max_shape: Tuple[int, ...] = None,
        chunks=None,
        compressor="lz4",
    ):
        """Constructs a Polygon HubSchema.
        Args:
        shape: tuple of ints or None, i.e (None, 2)

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
        """Check if provided shape  maches polygon characteristics."""
        if len(shape) != 2 or shape[-1] != 2:
            raise ValueError("Wrong polygon shape provided")

    def __str__(self):
        out = super().__str__()
        out = "Polygon" + out[6:]
        return out

    def __repr__(self):
        return self.__str__()
