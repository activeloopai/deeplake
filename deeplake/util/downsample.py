from typing import Optional, Union
from PIL import Image
import deeplake
from deeplake.core.partial_sample import PartialSample
import numpy as np
import io


def validate_downsampling(downsampling):
    if downsampling is None:
        return None, None
    if len(downsampling) != 2:
        raise ValueError(
            f"Downsampling must be a tuple of the form (downsampling_factor, number_of_layers), got {downsampling}"
        )
    downsampling_factor, number_of_layers = downsampling
    if downsampling_factor < 1:
        raise ValueError("Downsampling factor must be >= 1")
    if number_of_layers < 1:
        raise ValueError("Number of layers must be >= 1")
    return downsampling_factor, number_of_layers


def needs_downsampling(sample: Image.Image, factor: int):
    if sample.size[0] * sample.size[1] <= 100 * factor * factor:
        return False
    return sample.size[0] // factor > 0 and sample.size[1] // factor > 0


def get_filter(htype):
    if "image" in htype:
        return Image.BILINEAR
    if "mask" in htype:
        return Image.NEAREST
    raise ValueError(f"Got unexpected htype {htype}")


def downsample_sample(
    sample: Optional[Union[Image.Image, PartialSample]],
    factor: int,
    compression: Optional[str],
    htype: str,
):
    if isinstance(sample, PartialSample):
        return sample.downsample(factor)

    if sample is None or not needs_downsampling(sample, factor):
        return None

    downsampled_sample = sample.resize(
        (sample.size[0] // factor, sample.size[1] // factor), get_filter(htype)
    )
    if compression is None:
        return np.array(downsampled_sample)
    with io.BytesIO() as f:
        downsampled_sample.save(f, format=compression)
        image_bytes = f.getvalue()
        return deeplake.core.sample.Sample(buffer=image_bytes, compression=compression)


def get_downsample_factor(key: str):
    return int(key.split("_")[-1])


def apply_partial_downsample(tensor, global_sample_index, val):
    downsample_sub_index, new_value = val
    tensor[global_sample_index][downsample_sub_index] = new_value
