import numpy as np
from PIL import Image
from io import BytesIO

import hub
import hub.schema
from hub.utils import Timer

IMAGE_PATH = "./images/compression_benchmark_image.png"
IMG = Image.open(IMAGE_PATH)

REPEAT_TIMES = 100


def bench_pil_compression(times=REPEAT_TIMES):
    with Timer("PIL compression"):
        for i in range(times):
            b = BytesIO()
            IMG.save(b, format="png")


def bench_hub_compression(times=REPEAT_TIMES):
    arr = np.array(IMG)
    ds = hub.Dataset(
        "./data/bench_png_compression",
        mode="w",
        shape=times,
        schema={"image": hub.schema.Image(arr.shape, compressor="png")},
    )

    batch = np.zeros((times,) + arr.shape, dtype="uint8")
    for i in range(times):
        batch[i] = arr

    with Timer("Hub compression"):
        ds["image", :times] = batch


if __name__ == "__main__":
    bench_pil_compression()
    bench_hub_compression()
