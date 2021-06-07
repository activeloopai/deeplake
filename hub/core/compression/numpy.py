from io import BytesIO

import numpy as np

from hub.core.compression import BaseNumCodec


class NUMPY(BaseNumCodec):
    def __init__(self):
        super().__init__()

    def encode(self, array: np.ndarray) -> bytes:
        """
        Encode given array

        Example:
            arr = np.arange(100, 100, 2, dtype='uint8')
            arr_encoded = numpy_codec.encode(x)

        Args:
            array (np.ndarray): Data to be encoded

        Returns:
            Encoded data.
        """
        with BytesIO() as f:
            np.save(f, array, allow_pickle=True)
            return f.getvalue()

    def decode(self, bytes_: bytes) -> np.ndarray:
        """
        Decode data from buffer.

        Example:
            arr_decoded = numpy_codec.decode(arr_encoded)

        Args:
            bytes_ (bytes): Encoded data

        Returns:
            Decoded data.
        """
        with BytesIO(bytes_) as f:
            return np.load(f, allow_pickle=True)
