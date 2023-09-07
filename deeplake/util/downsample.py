from typing import Optional, Union
from PIL import Image  # type: ignore
import deeplake
from deeplake.core.tiling.sample_tiles import SampleTiles
from deeplake.core.partial_sample import PartialSample
from deeplake.core.linked_tiled_sample import LinkedTiledSample
import numpy as np
import io
from deeplake.core.tensor_link import read_linked_sample
import warnings


def validate_downsampling(downsampling):
    if downsampling is None:
        return None, None
    if len(downsampling) != 2:
        raise ValueError(
            f"Downsampling must be a tuple of the form (downsampling_factor, number_of_layers), got {downsampling}"
        )
    downsampling_factor, number_of_layers = downsampling
    if downsampling_factor < 1 or not isinstance(downsampling_factor, int):
        raise ValueError("Downsampling factor must be an integer >= 1")
    if number_of_layers < 1 or not isinstance(number_of_layers, int):
        raise ValueError("Number of layers must be an integer >= 1")

    return downsampling_factor, number_of_layers


def needs_downsampling(sample, factor: int):
    if isinstance(sample, Image.Image):
        dimensions = sample.size
    elif isinstance(sample, np.ndarray):
        dimensions = sample.shape
        if len(dimensions) == 3 and dimensions[2] == 0:
            return False
        dimensions = dimensions[:2]

    if dimensions[0] * dimensions[1] <= 100 * factor * factor:
        return False
    return dimensions[0] // factor > 0 and dimensions[1] // factor > 0


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
    partial: bool = False,
    link_creds=None,
):
    try:
        if sample is None:
            return None
        elif isinstance(sample, SampleTiles):
            sample = sample.arr or PartialSample(
                sample.sample_shape, sample.tile_shape, sample.dtype
            )

        if isinstance(sample, PartialSample):
            return sample.downsample(factor)
        elif isinstance(sample, LinkedTiledSample):
            return downsample_link_tiled(sample, factor, compression, htype, link_creds)

        if not partial and not needs_downsampling(sample, factor):
            arr = np.array(sample) if isinstance(sample, Image.Image) else sample
            required_shape = tuple([0] * len(arr.shape))
            required_dtype = arr.dtype
            return np.ones(required_shape, dtype=required_dtype)

        if isinstance(sample, np.ndarray):
            downsampled_sample = sample[::factor, ::factor]
            return downsampled_sample
        size = sample.size[0] // factor, sample.size[1] // factor
        downsampled_sample = sample.resize(size, get_filter(htype))
        if compression is None:
            return np.array(downsampled_sample)
        with io.BytesIO() as f:
            downsampled_sample.save(f, format=compression)  # type: ignore
            image_bytes = f.getvalue()
            return deeplake.core.sample.Sample(
                buffer=image_bytes, compression=compression
            )
    except Exception as e:
        if partial:
            raise e
        warnings.warn(f"Failed to downsample sample of type {type(sample)}")
        return None


def downsample_link_tiled(
    sample: LinkedTiledSample, factor, compression, htype, link_creds=None
):
    shape = sample.shape
    tile_shape = sample.tile_shape
    downsampled_tile_size = tile_shape[0] // factor, tile_shape[1] // factor
    downsampled_sample_size = shape[0] // factor, shape[1] // factor
    path_array = sample.path_array
    arr = None
    for i in range(path_array.shape[0]):
        for j in range(path_array.shape[1]):
            path = sample.path_array[i, j].flatten()[0]
            creds_key = sample.creds_key
            tile_pil = read_linked_sample(path, creds_key, link_creds, verify=False).pil

            # reverse the size because PIL expects (width, height)
            downsampled_tile_pil = tile_pil.resize(
                downsampled_tile_size[::-1], get_filter(htype)
            )
            downsampled_tile_arr = np.array(downsampled_tile_pil)
            if arr is None:
                arr_size = downsampled_sample_size + shape[2:]
                arr = np.zeros(arr_size, dtype=downsampled_tile_arr.dtype)
            arr[
                i * downsampled_tile_size[0] : (i + 1) * downsampled_tile_size[0],
                j * downsampled_tile_size[1] : (j + 1) * downsampled_tile_size[1],
            ] = downsampled_tile_arr

    if compression is None:
        return arr
    with io.BytesIO() as f:
        Image.fromarray(arr).save(f, format=compression)
        image_bytes = f.getvalue()
        return deeplake.core.sample.Sample(buffer=image_bytes, compression=compression)


def apply_partial_downsample(tensor, global_sample_index, val):
    downsample_sub_index, new_value = val
    tensor[global_sample_index][downsample_sub_index] = new_value
