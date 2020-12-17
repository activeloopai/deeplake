from PIL import Image
import numpy as np
from io import BytesIO

from hub.utils import Timer
import hub
import hub.schema


img_path = "./benchmarks/sample.png"
count = 20


def bench_pil_compression(img_path=img_path, count=count):
    img = Image.open(img_path)
    arr = np.array(img)
    print(arr.shape)
    with Timer("PIL compression"):
        for i in range(0, count):
            img = Image.fromarray(arr)
            b = BytesIO()
            img.save(b, format="png")
            assert b.tell() > 0


def bench_hub_compression(img_path=img_path, count=count):
    img = Image.open(img_path)
    arr = np.array(img)
    print(arr.shape)
    ds = hub.Dataset(
        "./data/benchmarks/bench_png_compression",
        mode="w",
        shape=count,
        schema={"image": hub.schema.Image(arr.shape, compressor="png")},
    )
    print(ds._tensors["/image"].chunks)
    bigarr = np.zeros((count,) + arr.shape, dtype="uint8")
    for i in range(count):
        bigarr[i] = arr

    with Timer("Hub compression"):
        ds["image", :count] = bigarr
        # for i in range(count):
        #     ds["image", i] = arr


if __name__ == "__main__":
    bench_pil_compression()
    bench_hub_compression()
