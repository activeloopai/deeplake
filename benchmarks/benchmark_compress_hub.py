import numpy as np
from PIL import Image

import hub_v1


def benchmark_compress_hub_setup(
    times, image_path="./images/compression_benchmark_image.png"
):
    img = Image.open(image_path)
    arr = np.array(img)
    ds = hub_v1.Dataset(
        "./data/bench_png_compression",
        mode="w",
        shape=times,
        schema={"image": hub_v1.schema.Image(arr.shape, compressor="png")},
    )

    batch = np.zeros((times,) + arr.shape, dtype="uint8")
    for i in range(times):
        batch[i] = arr

    return (ds, times, batch)


def benchmark_compress_hub_run(params):
    ds, times, batch = params
    ds["image", :times] = batch
