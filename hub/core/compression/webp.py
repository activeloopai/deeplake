from io import BytesIO

import numcodecs  # type: ignore
import numpy as np
from PIL import Image, ImageOps  # type: ignore

from hub.core.compression import BaseImgCodec


class WEBP(BaseImgCodec):
    def __init__(self, **kwargs):
        """
        Initialize WebP compressor.

        Args:
            **kwargs: Optional; single_channel (bool): if True,
                encoder will remove the last dimension of input if it is 1.
                quality (int): The image quality, on a scale from 1 (worst) to 95 (best). Default: 95.
                    For large images set quality <= 85.

        Raises:
            ValueError: If keyword arguments contain not supported arguments.
        """
        super().__init__()
        self.codec_id = "webp"
        webp_args = {"single_channel": True, "quality": 95}

        diff = set(kwargs.keys()) - set(webp_args.keys())
        if diff:
            raise ValueError("Invalid args:", tuple(diff))

        self.single_channel = kwargs.get("single_channel", webp_args["single_channel"])
        self.quality = kwargs.get("quality", webp_args["quality"])

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
            img = Image.fromarray(image)
            img = img.convert("RGB")
            img.save(buffer, format=self.codec_id, quality=self.quality)
            return buffer.getvalue()

    def decode_single_image(self, buf: bytes, image_shape: tuple) -> np.ndarray:
        """
        Decode single image from buffer.

        Example:
            imgs_decoded = png_codec.decode(imgs_encoded)

        Args:
            buf (bytes): Encoded image
            image_shape (tuple): Shape of encoded image.

        Returns:
            Decoded data.
        """
        with BytesIO(buf) as buffer:
            buffer.seek(0)
            image = Image.open(buffer, mode="r")
            if len(image_shape) == 2:
                return np.array(ImageOps.grayscale(image))
            return np.array(image)

    @classmethod
    def from_config(cls, config):
        """
        Create WEBP compressor from configuration dict.

        Args:
            config (Dict): Dictionary with compressor parameters.

        Returns:
            Compressor object with given parameters.
        """
        return WEBP(single_channel=config["single_channel"])


numcodecs.register_codec(WEBP, "webp")
