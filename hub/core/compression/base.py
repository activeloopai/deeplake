from typing import Union
from abc import ABC, abstractmethod  # type: ignore
import numcodecs  # type: ignore
from numcodecs.abc import Codec  # type: ignore

import numpy as np


class BaseNumCodec(ABC):
    """Base class for numcodec compressors"""

    @abstractmethod
    def encode(self, input: np.ndarray) -> bytes:
        pass

    @abstractmethod
    def decode(self, bytes: bytes) -> Union[np.ndarray, bytes]:
        pass

    @property
    @abstractmethod
    def __name__(self):
        pass


class BaseImgCodec(ABC, Codec):
    """Base class for image codecs"""

    def __init__(self, single_channel: bool = True) -> None:
        self.single_channel = single_channel
        self._msgpack = numcodecs.MsgPack()

    @property
    def __name__(self):
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
                (N, M, 3). In case of multiple images they should be stacked across the first axis.

        Raises:
            ValueError: If the shape length of input array is
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
            raise ValueError(
                f"The shape length {len(arr.shape)} of the given array should "
                f"be greater than the number of expected dimensions {shape_dims}"
            )
        if len(arr.shape) == shape_dims:
            return self._msgpack.encode(
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
        pass_shape = True if self.__name__ == "webp" else False
        if "items_shape" not in data:
            if pass_shape:
                images = self.decode_single_image(data["items"], data["image_shape"])
            else:
                images = self.decode_single_image(data["items"])
        else:
            items = data["items"]
            images = np.zeros(
                data["items_shape"] + data["image_shape"], dtype=data["dtype"]
            )

            for i, index in enumerate(np.ndindex(tuple(data["items_shape"]))):
                if pass_shape:
                    images[index] = self.decode_single_image(
                        items[i], data["image_shape"]
                    )
                else:
                    images[index] = self.decode_single_image(items[i])

        if data.get("append_one"):
            images = np.reshape(images, images.shape + (1,))
        return images

    def get_config(self):
        """Get compressor configuration dict"""
        return {"id": self.codec_id, "single_channel": self.single_channel}
