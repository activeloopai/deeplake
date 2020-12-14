from io import BytesIO

import zarr
import numcodecs
from numcodecs.abc import Codec
import numpy as np
from PIL import Image


class PngCodec(Codec):
    def __init__(self, solo_channel=True):
        self.codec_id = "png"
        self.solo_channel = solo_channel
        self._msgpack = numcodecs.MsgPack()

    def encode_single_image(self, image: np.ndarray) -> bytes:
        with BytesIO() as buffer:
            Image.fromarray(image).save(buffer, format="png")
            return buffer.getvalue()

    def decode_single_image(self, buf) -> np.ndarray:
        with BytesIO(buf) as buffer:
            buffer.seek(0)
            return np.array(Image.open(buffer, mode="r"))

    def encode(self, buf: np.ndarray):
        append_one = False
        if self.solo_channel and buf.shape[-1] == 1:
            buf = np.reshape(buf, buf.shape[:-1])
            append_one = True
        if not self.solo_channel or append_one:
            shape_dims = 2
        else:
            shape_dims = 3
        assert len(buf.shape) >= shape_dims
        if len(buf.shape) == shape_dims:
            return self._msgpack.encode(
                [{"items": self.encode_single_image(buf), "append_one": append_one}]
            )
        else:
            image_shape = buf.shape[-shape_dims:]
            items_shape = buf.shape[:-shape_dims]
            items = []
            for i in np.ndindex(items_shape):
                items.append(self.encode_single_image(buf[i]))
            return self._msgpack.encode(
                [
                    {
                        "items": items,
                        "items_shape": items_shape,
                        "image_shape": image_shape,
                        "dtype": str(buf.dtype),
                        "append_one": append_one,
                    }
                ]
            )

    def decode(self, buf, out=None):
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
        return {"id": self.codec_id, "solo_channel": self.solo_channel}

    # def __dict__(self):
    #     return self.get_config()

    @classmethod
    def from_config(cls, config):
        return PngCodec(config["solo_channel"])


numcodecs.register_codec(PngCodec, "png")
