from io import BytesIO
import numcodecs
from numcodecs.abc import Codec

import numpy as np
from PIL import Image


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


class BaseImgCodec(Codec):
    """Base class for image codecs"""

    def __init__(self, single_channel: bool = True):
        self.single_channel = single_channel
        self._msgpack = numcodecs.MsgPack()

    def encode_single_image():
        return NotImplementedError()

    def decode_single_image():
        return NotImplementedError()

    def encode(self, arr: np.ndarray):
        """
        Encode array with one or multiple images.

        Example:
            img = np.ones((4, 100, 100, 1))
            encoding = codec.encode(img)

        Args:
            array (np.ndarray): Image data to be encoded. Can contain image/s with shapes:
            (N, M) if image has one channel and single_channel is False,
            (N, M, 1) if image has one channel and single_channel is True,
            (N, M, 3).
            In case of multiple images they should be stacked across the first axis.

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
        assert len(arr.shape) >= shape_dims
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

    def decode(self, buf, out=None):
        """
        Decode images from buffer.

        Example:
            images_decoded = codec.decode(images_encoded)

        Args:
            bytes_ (bytes): Encoded images and metadata.

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

    def __init__(self, single_channel: bool = True):
        """
        Initialize Jpeg compressor

        Args:
            single_channel (bool): if True, encoder will remove the last dimension of input if it is 1.
        """
        super().__init__()
        self.codec_id = "jpeg"
        self.single_channel = single_channel

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
                buffer, format=self.codec_id, subsampling=0, quality=90
            )
            return buffer.getvalue()

    def decode_single_image(self, buf) -> np.ndarray:
        """
        Decode single image from buffer.

        Example:
            imgs_decoded = jpeg_codec.decode(imgs_encoded)

        Args:
            bytes_ (bytes): Encoded image

        Returns:
            Decoded data.
        """
        with BytesIO(buf) as buffer:
            buffer.seek(0)
            return np.array(Image.open(buffer, mode="r"))

    @classmethod
    def from_config(cls, config):
        return JpegCodec(config["single_channel"])


class PngCodec(BaseImgCodec, Codec):
    def __init__(self, single_channel: bool = True):
        """
        Initialize PNG compressor

        Args:
            single_channel (bool): if True, encoder will remove the last dimension of input if it is 1.
        """
        super().__init__()
        self.codec_id = "png"
        self.single_channel = single_channel

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

    def decode_single_image(self, buf) -> np.ndarray:
        """
        Decode single image from buffer.

        Example:
            imgs_decoded = png_codec.decode(imgs_encoded)

        Args:
            bytes_ (bytes): Encoded image

        Returns:
            Decoded data.
        """
        with BytesIO(buf) as buffer:
            buffer.seek(0)
            return np.array(Image.open(buffer, mode="r"))

    @classmethod
    def from_config(cls, config):
        return PngCodec(config["single_channel"])


numcodecs.register_codec(PngCodec, "png")
numcodecs.register_codec(JpegCodec, "jpeg")
