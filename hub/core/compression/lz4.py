from typing import Union

import msgpack  # type: ignore
import numcodecs  # type: ignore
import numpy as np

from hub.core.compression import BaseNumCodec
from hub.core.compression.constants import MSGPACK


class LZ4(BaseNumCodec):
    def __init__(self, **kwargs):
        """
        Initialize Lz4 compressor.

        Args:
            **kwargs: Optional; acceleration (int): Acceleration level.
                The larger the acceleration value, the faster the algorithm, but also the lesser the compression.
                For more information see: http://fastcompression.blogspot.com/2015/04/sampling-or-faster-lz4.html

        Raises:
            ValueError: If ketword arguments contain not supported arguments.
        """
        if kwargs and "acceleration" not in kwargs:
            raise ValueError("Invalid args:", kwargs.keys())
        acceleration = kwargs.get("acceleration", numcodecs.lz4.DEFAULT_ACCELERATION)
        self.compressor = numcodecs.lz4.LZ4(acceleration)

    def encode(self, input: Union[np.ndarray, bytes]) -> bytes:
        """
        Encode given array

        Example:
            arr = np.arange(100, 100, 2, dtype='uint8')
            arr_encoded = lz4_codec.encode(x)

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
            arr_decoded = lz4_codec.decode(arr_encoded)

        Args:
            bytes_ (bytes): Encoded data

        Returns:
            Decoded data.
        """
        try:
            data = MSGPACK.decode(bytes_)[0]
        except (msgpack.exceptions.ExtraData, ValueError):
            return self.compressor.decode(bytes_)
        decoded_buf = self.compressor.decode(data["item"])
        arr = np.frombuffer(decoded_buf, dtype=np.dtype(data["dtype"]))
        arr = arr.reshape(data["shape"])
        return arr
