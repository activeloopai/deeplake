from typing import Union

import msgpack  # type: ignore
import numcodecs  # type: ignore
import numpy as np

from hub.core.compression import BaseNumCodec
from hub.core.compression.constants import MSGPACK


class ZSTD(BaseNumCodec):
    def __init__(self, **kwargs):
        """
        Initialize Zstd compressor

        Args:
            **kwargs: Optional; level (int): Compression level (1-22).

        Raises:
            ValueError: If ketword arguments contain not supported arguments.
        """
        if kwargs and "level" not in kwargs:
            raise ValueError("Invalid args:", kwargs.keys())
        level = kwargs.get("level", numcodecs.zstd.DEFAULT_CLEVEL)
        self.compressor = numcodecs.zstd.Zstd(level)

    def encode(self, input: Union[np.ndarray, bytes]) -> bytes:
        """
        Encode given array

        Example:
            arr = np.arange(100, 100, 2, dtype='uint8')
            arr_encoded = zstd_codec.encode(x)

        Args:
            input (np.ndarray/bytes): Data to be encoded

        Returns:
            Encoded data.
        """
        if isinstance(input, bytes):
            return self.compressor.encode(input)
        return MSGPACK.encode(
            [
                {
                    "item": self.compressor.encode(input),
                    "dtype": input.dtype.name,
                    "shape": input.shape,
                }
            ]
        )

    def decode(self, bytes_: bytes) -> Union[np.ndarray, bytes]:
        """
        Decode data from buffer.

        Example:
            arr_decoded = zstd_codec.decode(arr_encoded)

        Args:
            bytes_ (bytes): Encoded data

        Returns:
            Decoded data.
        """
        try:
            data = MSGPACK.decode(bytes_)[0]
        except msgpack.exceptions.ExtraData:
            return self.compressor.decode(bytes_)
        decoded_buf = self.compressor.decode(data["item"])
        arr = np.frombuffer(decoded_buf, dtype=np.dtype(data["dtype"]))
        arr = arr.reshape(data["shape"])
        return arr
