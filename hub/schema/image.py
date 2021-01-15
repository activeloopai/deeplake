from typing import Tuple

import numpy as np

from hub.schema.features import Tensor


class Image(Tensor):
    """| `HubSchema` for images.

    Output: `tf.Tensor` of type `tf.uint8` and shape `[height, width, num_channels]`
    for BMP, JPEG, and PNG images

    Example:
    ----------
    >>> image_tensor = Image(shape=(None, None, 1),
    >>>                      encoding_format='png')
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (None, None, 3),
        dtype="uint8",
        # TODO Add back encoding_format (probably named compress) when support for png and jpg support will be added
        max_shape: Tuple[int, ...] = None,
        chunks=None,
        compressor="lz4",
    ):
        """| Construct the connector.

        Parameters
        ----------
        shape: tuple of ints or None
            The shape of decoded image:
            (height, width, channels) where height and width can be None.
            Defaults to (None, None, 3).
        dtype: `uint16` or `uint8` (default)
            `uint16` can be used only with png encoding_format
        encoding_format: 'jpeg' or 'png' (default)
             Format to serialize np.ndarray images on disk.
        max_shape : Tuple[int]
            Maximum shape of tensor shape if tensor is dynamic
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks


        Returns
        ----------
        `tf.Tensor` of type `tf.uint8` and shape `[height, width, num_channels]`
        for BMP, JPEG, and PNG images

        Raises
        ----------
        ValueError: If the shape, dtype or encoding formats are invalid

        """
        self._set_dtype(dtype)
        super().__init__(
            shape,
            dtype,
            max_shape=max_shape,
            chunks=chunks,
            compressor=compressor,
        )

    def _set_dtype(self, dtype):
        """Set the dtype."""
        dtype = str(np.dtype(dtype))
        if dtype not in ("uint8", "uint16"):
            raise ValueError(f"Not supported dtype for {self.__class__.__name__}")
        self.dtype = dtype

    def __str__(self):
        out = super().__str__()
        out = "Image" + out[6:]
        return out

    def __repr__(self):
        return self.__str__()
