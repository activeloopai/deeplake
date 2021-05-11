from io import BytesIO
import numcodecs  # type: ignore
from numcodecs.abc import Codec  # type: ignore

import numpy as np
from PIL import Image  # type: ignore


class BaseImgCodec(Codec):
    """Base class for image codecs"""

    def __init__(self, single_channel: bool = True):
        self.single_channel = single_channel
        self._msgpack = numcodecs.MsgPack()

    def encode_single_image(self, image: np.ndarray) -> bytes:
        raise NotImplementedError()

    def decode_single_image(self, buf: bytes) -> np.ndarray:
        raise NotImplementedError()

    def encode(self, arr: np.ndarray) -> bytes:
        """
        Encode array with one or multiple images.

        Example:
            img = np.ones((4, 100, 100, 1))
            encoding = codec.encode(img)

        Args:
            arr (np.ndarray): Image data to be encoded. Can contain image/s with shapes:
            (N, M) if image has one channel and single_channel is False,
            (N, M, 1) if image has one channel and single_channel is True,
            (N, M, 3).
            In case of multiple images they should be stacked across the first axis.

        Raises:
            ValueError: If yhe shape length of input array is
            less than the number of expected dimensions"

        Returns:
            Encoded dictionary of images and metadata.
        """
        append_one = False
        if self.single_channel and arr.shape[-1] == 1:
            arr = np.reshape(arr, arr.shape[:-1])
            append_one = True
        if not self.single_channel or append_one:
            shape_dims = 2
        else:
            shape_dims = 3
        if len(arr.shape) < shape_dims:
            raise ValueError(
                f"The shape length {len(arr.shape)} of the given array should "
                f"be greater than the number of expected dimensions {shape_dims}"
            )
        if len(arr.shape) == shape_dims:
            return self._msgpack.encode(
                [{"items": self.encode_single_image(arr), "append_one": append_one}]
            )
        else:
            image_shape = arr.shape[-shape_dims:]
            items_shape = arr.shape[:-shape_dims]
            items = []
            for i in np.ndindex(items_shape):
                items.append(self.encode_single_image(arr[i]))
            return self._msgpack.encode(
                [
                    {
                        "items": items,
                        "items_shape": items_shape,
                        "image_shape": image_shape,
                        "dtype": str(arr.dtype),
                        "append_one": append_one,
                    }
                ]
            )

    def decode(self, buf: bytes) -> np.ndarray:
        """
        Decode images from buffer.

        Example:
            images_decoded = codec.decode(images_encoded)

        Args:
            buf (bytes): Encoded images and metadata.

        Returns:
            Decoded image or array with multiple images.
        """
        data = self._msgpack.decode(buf)[0]
        if "items_shape" not in data:
            images = self.decode_single_image(data["items"])
        else:
            items = data["items"]
            images = np.zeros(
                data["items_shape"] + data["image_shape"], dtype=data["dtype"]
            )

            for i, index in enumerate(np.ndindex(tuple(data["items_shape"]))):
                images[index] = self.decode_single_image(items[i])

        if data.get("append_one"):
            images = np.reshape(images, images.shape + (1,))
        return images

    def get_config(self):
        """Get compressor configuration dict"""
        return {"id": self.codec_id, "single_channel": self.single_channel}


class JpegCodec(BaseImgCodec, Codec):
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

    @property
    def __name__(self):
        return self.codec_id

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
        return JpegCodec(single_channel=config["single_channel"])


class PngCodec(BaseImgCodec, Codec):
    def __init__(self, **kwargs):
        """
        Initialize PNG compressor

        Args:
            **kwargs: Optional; single_channel (bool): if True,
            encoder will remove the last dimension of input if it is 1.

        Raises:
            ValueError: If ketword arguments contain not supported arguments.
        """
        super().__init__()
        self.codec_id = "png"
        if kwargs and "single_channel" not in kwargs:
            raise ValueError("Invalid args:", kwargs.keys())

        self.single_channel = kwargs.get("single_channel", True)

    @property
    def __name__(self):
        return "png"

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
        return PngCodec(single_channel=config["single_channel"])


numcodecs.register_codec(PngCodec, "png")
numcodecs.register_codec(JpegCodec, "jpeg")
