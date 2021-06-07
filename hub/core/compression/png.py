from io import BytesIO

import numcodecs  # type: ignore
import numpy as np
from PIL import Image  # type: ignore

from hub.core.compression import BaseImgCodec


class PNG(BaseImgCodec):
    def __init__(self, **kwargs):
        """
        Initialize PNG compressor.

        Args:
            **kwargs: Optional; single_channel (bool): if True,
                encoder will remove the last dimension of input if it is 1.

        Raises:
            ValueError: If keyword arguments contain not supported arguments.
        """
        super().__init__()
        self.codec_id = "png"
        if kwargs and "single_channel" not in kwargs:
            raise ValueError("Invalid args:", kwargs.keys())

        self.single_channel = kwargs.get("single_channel", True)

    def encode_single_image(self, image: np.ndarray) -> bytes:
        """
        Encode single image.

        Example:
            img = np.ones(100, 100, 3)
            img_encoded = png_codec.encode_single_image(img)

        Args:
            image (np.ndarray): Single image to be encoded

        Returns:
            Encoded data.
        """
        with BytesIO() as buffer:
            Image.fromarray(image).save(buffer, format=self.codec_id)
            return buffer.getvalue()

    def decode_single_image(self, buf: bytes) -> np.ndarray:
        """
        Decode single image from buffer.

        Example:
            imgs_decoded = png_codec.decode(imgs_encoded)

        Args:
            buf (bytes): Encoded image

        Returns:
            Decoded data.
        """
        with BytesIO(buf) as buffer:
            buffer.seek(0)
            return np.array(Image.open(buffer, mode="r"))

    @classmethod
    def from_config(cls, config):
        """
        Create PNG compressor from configuration dict.

        Args:
            config (Dict): Dictionary with compressor parameters.

        Returns:
            Compressor object with given parameters.
        """
        return PNG(single_channel=config["single_channel"])


numcodecs.register_codec(PNG, "png")
