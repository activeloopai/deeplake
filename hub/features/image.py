from typing import Tuple

from hub.features.features import Tensor


class Image(Tensor):
    """`FeatureConnector` for images
    Output:
    `tf.Tensor` of type `tf.uint8` and shape `[height, width, num_channels]`
    for BMP, JPEG, and PNG images

    Example:
    ```python
    image_tensor = Image(shape=(None, None, 1),
                                encoding_format='png')
    ```
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (None, None, 3),
        dtype="uint8",
        encoding_format: str = "png",
        max_shape: Tuple[int, ...] = None,
        chunks=True,
    ):
        """Construct the connector.
        Args:
        shape: tuple of ints or None, the shape of decoded image:
            (height, width, channels) where height and width can be None.
            Defaults to (None, None, 3).
        dtype: `uint16` or `uint8` (default).
            `uint16` can be used only with png encoding_format
        encoding_format: 'jpeg' or 'png' (default). Format to serialize np.ndarray
            images on disk.
        channels: list or tuple of the names of channels: ("nir", "red", "green",...)
        Raises:
        ValueError: If the shape, dtype or encoding formats are invalid
        """
        self._set_dtype(dtype)
        super(Image, self).__init__(shape, dtype, max_shape=max_shape, chunks=chunks)
        self._set_encoding_format(encoding_format)

    def _set_encoding_format(self, encoding_format):
        """Set the encoding format."""
        if encoding_format not in ("png", "jpeg"):
            raise ValueError("Not supported encoding format")
        self.encoding_format = encoding_format

    def _set_dtype(self, dtype):
        """Set the dtype."""
        if dtype not in ("uint8", "uint16"):
            raise ValueError(f"Not supported dtype for {self.__class__.__name__}")
        self.dtype = dtype

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
