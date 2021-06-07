from abc import ABC, abstractmethod  # type: ignore
from typing import Union, Dict

import numpy as np
from numcodecs.abc import Codec  # type: ignore

from hub.util.exceptions import InvalidImageDimensions
from .constants import WEBP_COMPRESSOR_NAME, MSGPACK


class BaseNumCodec(ABC):
    """Base class for numcodec compressors"""

    @abstractmethod
    def encode(self, input: np.ndarray) -> bytes:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, bytes: bytes) -> Union[np.ndarray, bytes]:
        raise NotImplementedError()

    @property
    def __name__(self):
        return self.__class__.__name__.lower()


class BaseImgCodec(ABC, Codec):
    """Base class for image codecs"""

    def __init__(self, single_channel: bool = True) -> None:
        self.single_channel = single_channel

    @property
    def __name__(self):
        return self.__class__.__name__.lower()

    def encode(self, arr: np.ndarray) -> bytes:
        """
        Encode array with one or multiple images.

        Example:
            img = np.ones((4, 100, 100, 1))
            encoded_img = codec.encode(img)

        Args:
            arr (np.ndarray): Image data to be encoded. Can contain image/s with shapes:
                (N, M) if image has one channel and single_channel is False,
                (N, M, 1) if image has one channel and single_channel is True,
                (N, M, 3) if image has multiple channels.
                In case of multiple images they should be stacked across the first axis.

        Raises:
            InvalidImageDimensions: If the shape length of input array is
                less than the number of expected dimensions.

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
            raise InvalidImageDimensions(len(arr.shape), shape_dims)
        if len(arr.shape) == shape_dims:
            return MSGPACK.encode(
                [
                    {
                        "items": self.encode_single_image(arr),
                        "append_one": append_one,
                        "image_shape": arr.shape,
                    }
                ]
            )
        else:
            image_shape = arr.shape[-shape_dims:]
            items_shape = arr.shape[:-shape_dims]
            items = []
            for i in np.ndindex(items_shape):
                items.append(self.encode_single_image(arr[i]))
            return MSGPACK.encode(
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

    def decode_data_single_image(self, data: Dict, pass_shape: bool):
        """
        Decode data that contains only one image

        Args:
            data (Dict): Dictionary of encoded image
            pass_shape (bool): Use encoded image shape for decoding.
                Applied only for WebP compression.

        Returns:
            np.ndarray of decoded image
        """
        if pass_shape:
            images = self.decode_single_image(data["items"], data["image_shape"])
        else:
            images = self.decode_single_image(data["items"])
        return images

    def decode_multiple_images(self, data: Dict, pass_shape: bool):
        """
        Decode data that contains multiple images

        Args:
            data (Dict): Dictionary of encoded images
            pass_shape (bool): Use encoded shapes for decoding.
                Applied only for WebP compression.

        Returns:
            np.ndarray of decoded images
        """
        items = data["items"]
        images = np.zeros(
            data["items_shape"] + data["image_shape"], dtype=data["dtype"]
        )

        for i, index in enumerate(np.ndindex(tuple(data["items_shape"]))):
            if pass_shape:
                images[index] = self.decode_single_image(items[i], data["image_shape"])
            else:
                images[index] = self.decode_single_image(items[i])
        return images

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
        data = MSGPACK.decode(buf)[0]
        pass_shape = True if self.__name__ == WEBP_COMPRESSOR_NAME else False
        if "items_shape" not in data:
            images = self.decode_data_single_image(data, pass_shape)
        else:
            images = self.decode_multiple_images(data, pass_shape)

        if data.get("append_one"):
            images = np.reshape(images, images.shape + (1,))
        return images

    def get_config(self):
        """Get compressor configuration dict"""
        return {"id": self.codec_id, "single_channel": self.single_channel}
