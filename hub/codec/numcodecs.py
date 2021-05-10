from io import BytesIO
import numcodecs

import numpy as np


class Base:
    """Base class for numcodec compressors"""

    def encode(self, array: np.ndarray) -> bytes:
        raise NotImplementedError()

    def decode(self, bytes: bytes) -> np.ndarray:
        raise NotImplementedError()


class NumPy(Base):
    """Numpy compressor"""

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


class Lz4(Base):
    """Lz4 compressor"""

    def __init__(self, acceleration: int):
        """
        Initialize Lz4 compressor

        Args:
            acceleration (int): Acceleration level.
            The larger the acceleration value, the faster the algorithm, but also the lesser the compression.
        """
        self.compressor = numcodecs.lz4.LZ4(acceleration)
        self._msgpack = numcodecs.MsgPack()

    def encode(self, array: np.ndarray) -> bytes:
        """
        Encode given array

        Example:
            arr = np.arange(100, 100, 2, dtype='uint8')
            arr_encoded = lz4_codec.encode(x)

        Args:
            array (np.ndarray): Data to be encoded

        Returns:
            Encoded data.
        """
        return self._msgpack.encode(
            [
                {
                    "item": self.compressor.encode(array),
                    "dtype": array.dtype.name,
                    "shape": array.shape,
                }
            ]
        )

    def decode(self, bytes_: bytes) -> np.ndarray:
        """
        Decode data from buffer.

        Example:
            arr_decoded = lz4_codec.decode(arr_encoded)

        Args:
            bytes_ (bytes): Encoded data

        Returns:
            Decoded data.
        """
        data = self._msgpack.decode(bytes_)[0]
        decoded_buf = self.compressor.decode(data["item"])
        arr = np.frombuffer(decoded_buf, dtype=np.dtype(data["dtype"]))
        arr = arr.reshape(data["shape"])
        return arr


class Zstd(Base):
    """Zstd compressor"""

    def __init__(self, level: int):
        """
        Initialize Zstd compressor

        Args:
            level (int): Compression level (1-22).
        """
        self.compressor = numcodecs.zstd.Zstd(level)
        self._msgpack = numcodecs.MsgPack()

    def encode(self, array: np.ndarray) -> bytes:
        """
        Encode given array

        Example:
            arr = np.arange(100, 100, 2, dtype='uint8')
            arr_encoded = zstd_codec.encode(x)

        Args:
            array (np.ndarray): Data to be encoded

        Returns:
            Encoded data.
        """
        return self._msgpack.encode(
            [
                {
                    "item": self.compressor.encode(array),
                    "dtype": array.dtype.name,
                    "shape": array.shape,
                }
            ]
        )

    def decode(self, bytes_: bytes) -> np.ndarray:
        """
        Decode data from buffer.

        Example:
            arr_decoded = zstd_codec.decode(arr_encoded)

        Args:
            bytes_ (bytes): Encoded data

        Returns:
            Decoded data.
        """
        data = self._msgpack.decode(bytes_)[0]
        decoded_buf = self.compressor.decode(data["item"])
        arr = np.frombuffer(decoded_buf, dtype=np.dtype(data["dtype"]))
        arr = arr.reshape(data["shape"])
        return arr
