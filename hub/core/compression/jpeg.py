from io import BytesIO

import numcodecs  # type: ignore
import numpy as np
from PIL import Image  # type: ignore

from hub.core.compression import BaseImgCodec


class JPEG(BaseImgCodec):
    """Jpeg compressor for image data"""

    def __init__(self, **kwargs):
        """
        Initialize Jpeg compressor.

        Args:
            **kwargs: Optional; single_channel (bool): if True,
                encoder will remove the last dimension of input if it is 1.
                quality (int): The image quality, on a scale from 1 (worst) to 95 (best). Default: 95.

        Raises:
            ValueError: If ketword arguments contain not supported arguments.
        """
        super().__init__()
        self.codec_id = "jpeg"
        jpeg_args = {"single_channel": True, "quality": 95}

        diff = set(kwargs.keys()) - set(jpeg_args.keys())
        if diff:
            raise ValueError("Invalid args:", tuple(diff))

        self.single_channel = kwargs.get("single_channel", jpeg_args["single_channel"])
        self.quality = kwargs.get("quality", jpeg_args["quality"])

    def encode_single_image(self, image: np.ndarray) -> bytes:
        """
        Encode single image.

        Example:
            img = np.ones(100, 100, 3)
            img_encoded = jpeg_codec.encode_single_image(img)

        Args:
            image (np.ndarray): Single image to be encoded

        Returns:
            Encoded data.
        """
        with BytesIO() as buffer:
            Image.fromarray(image).save(
                buffer, format=self.codec_id, subsampling=0, quality=self.quality
            )
            return buffer.getvalue()

    def decode_single_image(self, buf: bytes) -> np.ndarray:
        """
        Decode single image from buffer.

        Example:
            imgs_decoded = jpeg_codec.decode(imgs_encoded)

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
        Create JPEG compressor from configuration dict.

        Args:
            config (Dict): Dictionary with compressor parameters.

        Returns:
            Compressor object with given parameters.
        """
        return JPEG(single_channel=config["single_channel"])


numcodecs.register_codec(JPEG, "jpeg")
