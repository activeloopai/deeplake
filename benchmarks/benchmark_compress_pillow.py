from PIL import Image
from io import BytesIO


def benchmark_compress_pillow_setup(times, image_path="./images/compression_benchmark_image.png"):
    img = Image.open(image_path)
    return (img, times)


def benchmark_compress_pillow_run(params):
    img, times = params
    for _ in range(times):
        b = BytesIO()
        img.save(b, format="png")
