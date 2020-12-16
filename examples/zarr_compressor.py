from io import BytesIO

import zarr
import numcodecs
from numcodecs.abc import Codec
from numcodecs import MsgPack
import numpy as np
from PIL import Image

from hub.utils import Timer
from hub.numcodecs import PngCodec


def main():
    numcodecs.register_codec(PngCodec, "png")
    with Timer("Compress"):
        arr = zarr.create(
            shape=(10, 10, 1920, 1080, 7),
            dtype="uint8",
            compressor=PngCodec(solo_channel=True),
            store=zarr.MemoryStore(),
        )
        arr[:] = np.ones((10, 10, 1920, 1080, 7), dtype="uint8")
        print(arr[:].shape)


if __name__ == "__main__":
    main()
