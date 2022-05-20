from io import BytesIO
from PIL import Image  # type: ignore
from hub.core.sample import Sample
import numpy as np


def convert_sample(image_sample: Sample, mode: str) -> Sample:
    if image_sample.path:
        image = Image.open(image_sample.path)
    elif image_sample._buffer:
        image = Image.open(BytesIO(image_sample._buffer))
    if image.mode == mode:
        image.close()
        return image_sample

    image = image.convert(mode)
    image_bytes = BytesIO()
    image.save(image_bytes, format=image_sample.compression)
    converted = Sample(
        buffer=image_bytes.getvalue(), compression=image_sample.compression
    )
    image.close()
    return converted


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    transform = np.array([[[299 / 1000, 587 / 1000, 114 / 1000]]])
    gray = np.sum(transform * arr[:, :, :3], axis=2, dtype=np.uint8)
    return gray


def convert_img_arr(image_arr: np.ndarray, mode: str) -> np.ndarray:
    if len(image_arr.shape) == 2:
        if mode == "L":
            return image_arr
        elif mode == "RGB":
            return np.tile(image_arr[:, :, np.newaxis], (1, 1, 3))

    if (image_arr.shape[-1]) == 4:
        if mode == "L":
            return to_grayscale(image_arr)
        elif mode == "RGB":
            return image_arr[:, :, :3]

    elif (image_arr.shape[-1]) == 3:
        if mode == "L":
            return to_grayscale(image_arr)
        elif mode == "RGB":
            return image_arr

    raise ValueError("Invalid image")
