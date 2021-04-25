"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
from typing import List, Union, Tuple
import cv2
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    Transpose,
    Blur,
    GaussNoise,
    RandomBrightnessContrast,
)
from PIL import Image
import hub
from hub import Dataset
from hub.api.datasetview import DatasetView


def horizontal_flip(
    ds: Union[Dataset, DatasetView],
    keys: Union[List[str], Tuple[str]] = ["image"],
    p: float = 1.0,
    scheduler: str = "threaded",
    workers: int = os.cpu_count() - 1,
):
    """Generic transform for horizontal flip
    Parameters
    ----------
    ds: Dataset or DatasetView
        hub.Dataset on which horizontal flip should be applied.
    keys: List or Tuple of str. Default: ['image']
        Apply horizontal flip only on specified keys from Dataset schema.
        Supproted input types: uint8, float32
    schduler: str. Default: 'threaded'.
        Use transform with this scheduler. Choices: ['threaded', 'ray_generator']
    p: float
        Probability of applying the transform. Default: 1.0

    Returns
    ----------
    hub.Dataset with transformed samples

    | Usage:
    ----------
    >>> ds = hub.Dataset("/my/dataset")
    >>> transformed_ds = horizontal_flip(ds, ["img"])
    >>> transformed_ds.store("./transformed_ds")
    """
    if isinstance(ds.schema, dict):
        schema_dict = ds.schema
    else:
        schema_dict = ds.schema.dict_

    @hub.transform(schema=schema_dict, scheduler=scheduler, workers=workers)
    def run_flip(sample, keys):
        for key in keys:
            if sample[key].dtype == "bool":
                sample[key] = sample[key].astype("uint8")
            sample[key] = HorizontalFlip(p=p)(image=sample[key])["image"]
        return sample

    return run_flip(ds, keys=keys)


def vertical_flip(
    ds: Union[Dataset, DatasetView],
    keys: Union[List, Tuple] = ["image"],
    scheduler: str = "threaded",
    workers: int = os.cpu_count() - 1,
    p: float = 1.0,
):
    """Generic transform for vertical flip
    Parameters
    ----------
    ds: Dataset or DatasetView
        hub.Dataset on which vertical flip should be applied.
    keys: List or Tuple of str. Default: ['image']
        Apply vertical flip only on specified keys from Dataset schema.
        Supproted input types: uint8, float32
    schduler: str. Default: 'threaded'.
        Use transform with this scheduler. Choices: ['threaded', 'ray_generator']
    p: float
        Probability of applying the transform. Default: 1.0

    Returns
    ----------
    hub.Dataset with transformed samples

    | Usage:
    ----------
    >>> ds = hub.Dataset("/my/dataset")
    >>> transformed_ds = vertical_flip(ds, ["img"])
    >>> transformed_ds.store("./transformed_ds")
    """
    if isinstance(ds.schema, dict):
        schema_dict = ds.schema
    else:
        schema_dict = ds.schema.dict_

    @hub.transform(schema=schema_dict, scheduler=scheduler, workers=workers)
    def run_flip(sample, keys):
        for key in keys:
            if sample[key].dtype == "bool":
                sample[key] = sample[key].astype("uint8")
            sample[key] = VerticalFlip(p=p)(image=sample[key])["image"]
        return sample

    return run_flip(ds, keys=keys)


def shift_scale_rotate(
    ds: Union[Dataset, DatasetView],
    keys: Union[List, Tuple] = ["image"],
    scheduler: str = "threaded",
    workers: int = os.cpu_count() - 1,
    p: float = 1.0,
    shift_limit=0.0625,
    scale_limit=0.1,
    rotate_limit=45,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
    mask_value=None,
    shift_limit_x=None,
    shift_limit_y=None,
):
    """Generic transform for shift, scale and rotate
    Parameters
    ----------
    ds: Dataset or DatasetView
        hub.Dataset on which shift_scale_rotate should be applied.
    keys: List or Tuple of str. Default: ['image']
        Apply shift_scale_rotate only on specified keys from Dataset schema.
        Supproted input types: uint8, float32
    schduler: str. Default: 'threaded'.
        Use transform with this scheduler. Choices: ['threaded', 'ray_generator']
    p: float
        Probability of applying the transform. Default: 1.0
    shift_limit: (float, float) or float. Default: (-0.0625, 0.0625).
        Shift factor range for both height and width. If shift_limit
        is a single float value, the range will be (-shift_limit, shift_limit).
        Absolute values for lower and upper bounds should lie in range [0, 1].
    scale_limit: (float, float) or float. Default: (-0.1, 0.1).
        Scaling factor range. If scale_limit is a single float value, the
        range will be (-scale_limit, scale_limit).
    rotate_limit: (int, int) or int. Default: (-45, 45).
        Rotation range. If rotate_limit is a single int value, the
        range will be (-rotate_limit, rotate_limit).
    interpolation: OpenCV flag. Default: cv2.INTER_LINEAR.
        Flag that is used to specify the interpolation algorithm.
        Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
        cv2.INTER_AREA, cv2.INTER_LANCZOS4.
    border_mode: OpenCV flag. Default: cv2.BORDER_REFLECT_101
        Flag that is used to specify the pixel extrapolation method.
        Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE,
        cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
    value: int, float, list of int, list of float. Default: None.
        Padding value if border_mode is cv2.BORDER_CONSTANT.
    mask_value (int, float, list of int, list of float). Default: None.
        Padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
    shift_limit_x: (float, float) or float.  Default: None.
        Shift factor range for width. If it is set then this value instead of
        shift_limit will be used for shifting width.
        If shift_limit_x is a single float value, the range will be
        (-shift_limit_x, shift_limit_x). Absolute values for lower and upper
        bounds should lie in the range [0, 1].
    shift_limit_y: (float, float) or float. Default: None.
        Shift factor range for height. If it is set then this value
        instead of shift_limit will be used for shifting height.
        If shift_limit_y is a single float value, the range will be (-shift_limit_y, shift_limit_y).
        Absolute values for lower and upper bounds should lie in the range [0, 1].
    Returns
    ----------
    hub.Dataset with transformed samples

    | Usage:
    ----------
    >>> ds = hub.Dataset("/my/dataset")
    >>> transformed_ds = shift_scale_rotate(ds, ["img"])
    >>> transformed_ds.store("./transformed_ds")
    """
    if isinstance(ds.schema, dict):
        schema_dict = ds.schema
    else:
        schema_dict = ds.schema.dict_

    @hub.transform(schema=schema_dict, scheduler=scheduler, workers=workers)
    def run_ssr(sample, keys):
        for key in keys:
            if sample[key].dtype == "bool":
                sample[key] = sample[key].astype("uint8")
            sample[key] = ShiftScaleRotate(
                p=p,
                shift_limit=shift_limit,
                scale_limit=scale_limit,
                rotate_limit=rotate_limit,
                interpolation=interpolation,
                border_mode=border_mode,
                value=value,
                mask_value=mask_value,
                shift_limit_x=shift_limit_x,
                shift_limit_y=shift_limit_y,
            )(image=sample[key])["image"]
        return sample

    return run_ssr(ds, keys=keys)


def blur(
    ds: Union[Dataset, DatasetView],
    keys: Union[List, Tuple] = ["image"],
    scheduler: str = "threaded",
    workers: int = os.cpu_count() - 1,
    p: float = 1.0,
    blur_limit: float = 7,
):
    """Generic transform for blur
    Parameters
    ----------
    ds: Dataset or DatasetView
        hub.Dataset on which blur transform should be applied.
    keys: List or Tuple of str. Default: ['image']
        Apply blur transform only on specified keys from Dataset schema.
        Supproted input types: uint8, float32
    schduler: str. Default: 'threaded'.
        Use transform with this scheduler. Choices: ['threaded', 'ray_generator']
    p: float
        Probability of applying the transform. Default: 1.
    blur_limit: int, (int, int). Default: (3, 7).
        Maximum kernel size for blurring the input image.
        Should be in range [3, inf)

    Returns
    ----------
    hub.Dataset with transformed samples

    | Usage:
    ----------
    >>> ds = hub.Dataset("/my/dataset")
    >>> transformed_ds = blur(ds, ["img"])
    >>> transformed_ds.store("./transformed_ds")
    """
    if isinstance(ds.schema, dict):
        schema_dict = ds.schema
    else:
        schema_dict = ds.schema.dict_

    @hub.transform(schema=schema_dict, scheduler=scheduler, workers=workers)
    def run_blur(sample, keys):
        for key in keys:
            if sample[key].dtype == "bool":
                sample[key] = sample[key].astype("uint8")
            sample[key] = Blur(p=p, blur_limit=blur_limit)(image=sample[key])["image"]
        return sample

    return run_blur(ds, keys=keys)


def random_brightness_contrast(
    ds: Union[Dataset, DatasetView],
    keys: Union[List, Tuple] = ["image"],
    scheduler: str = "threaded",
    workers: int = os.cpu_count() - 1,
    p: float = 1.0,
    brightness_limit=0.2,
    contrast_limit=0.2,
    brightness_by_max=True,
):
    """Generic transform for random brightness contrast
    Parameters
    ----------
    ds: Dataset or DatasetView
        hub.Dataset on which random brightness contrast transform should be applied.
    keys: List or Tuple of str. Default: ['image']
        Apply random brightness contrast transform only on specified keys from Dataset schema.
        Supproted input types: uint8, float32
    schduler: str. Default: 'threaded'.
        Use transform with this scheduler. Choices: ['threaded', 'ray_generator']
    p: float
        Probability of applying the transform. Default: 1.
    brightness_limit: (float, float) or float. Default: (-0.2, 0.2).
        Factor range for changing brightness.
        If limit is a single float, the range will be (-limit, limit).
    contrast_limit: (float, float) or float. Default: (-0.2, 0.2).
        Factor range for changing contrast.
        If limit is a single float, the range will be (-limit, limit).
    brightness_by_max: Boolean. Default: True.
        If True adjust contrast by image dtype maximum, else adjust contrast by image mean.

    Returns
    ----------
    hub.Dataset with transformed samples

    | Usage:
    ----------
    >>> ds = hub.Dataset("/my/dataset")
    >>> transformed_ds = random_brightness_contrast(ds, ["img"])
    >>> transformed_ds.store("./transformed_ds")
    """
    if isinstance(ds.schema, dict):
        schema_dict = ds.schema
    else:
        schema_dict = ds.schema.dict_

    @hub.transform(schema=schema_dict, scheduler=scheduler, workers=workers)
    def run_rbc(sample, keys):
        for key in keys:
            if sample[key].dtype == "bool":
                sample[key] = sample[key].astype("uint8")
            sample[key] = RandomBrightnessContrast(
                p=p,
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                brightness_by_max=brightness_by_max,
            )(image=sample[key])["image"]
        return sample

    return run_rbc(ds, keys=keys)


def gausse_noise(
    ds: Union[Dataset, DatasetView],
    keys: Union[List, Tuple] = ["image"],
    scheduler: str = "threaded",
    workers: int = os.cpu_count() - 1,
    p: float = 1.0,
    var_limit=(10.0, 50.0),
    mean=0,
):
    """Generic transform for adding gausse noise
    Parameters
    ----------
    ds: Dataset or DatasetView
        hub.Dataset on which gausse_noise transform should be applied.
    keys: List or Tuple of str. Default: ['image']
        Apply gausse_noise transform only on specified keys from Dataset schema.
        Supproted input types: uint8, float32
    schduler: str. Default: 'threaded'.
        Use transform with this scheduler. Choices: ['threaded', 'ray_generator']
    p: float
        Probability of applying the transform. Default: 1.
    var_limit: (float, float) or float. Default: (10.0, 50.0).
        Variance range for noise. If var_limit is a single float, the range
        will be (0, var_limit).
    mean: float. Default: 0
        Mean of the noise.
    Returns
    ----------
    hub.Dataset with transformed samples

    | Usage:
    ----------
    >>> ds = hub.Dataset("/my/dataset")
    >>> transformed_ds = gauss_noise(ds, ["img"])
    >>> transformed_ds.store("./transformed_ds")
    """
    if isinstance(ds.schema, dict):
        schema_dict = ds.schema
    else:
        schema_dict = ds.schema.dict_

    @hub.transform(schema=schema_dict, scheduler=scheduler, workers=workers)
    def run_gn(sample, keys):
        for key in keys:
            if sample[key].dtype == "bool":
                sample[key] = sample[key].astype("uint8")
            sample[key] = GaussNoise(p=p, var_limit=var_limit, mean=mean)(
                image=sample[key]
            )["image"]
        return sample

    return run_gn(ds, keys=keys)


def transpose(
    ds: Union[Dataset, DatasetView],
    keys: Union[List, Tuple] = ["image"],
    scheduler: str = "threaded",
    workers: int = os.cpu_count() - 1,
    p: float = 1.0,
):
    """Generic transform for transpose
    Parameters
    ----------
    ds: Dataset or DatasetView
        hub.Dataset on which transpose transform should be applied.
    keys: List or Tuple of str. Default: ['image']
        Apply transpose transform only on specified keys from Dataset schema.
        Supproted input types: uint8, float32
    schduler: str. Default: 'threaded'.
        Use transform with this scheduler. Choices: ['threaded', 'ray_generator']
    p: float
        Probability of applying the transform. Default: 1.

    Returns
    ----------
    hub.Dataset with transformed samples

    | Usage:
    ----------
    >>> ds = hub.Dataset("/my/dataset")
    >>> transformed_ds = transpose(ds, ["img"])
    >>> transformed_ds.store("./transformed_ds")
    """
    if isinstance(ds.schema, dict):
        schema_dict = ds.schema
    else:
        schema_dict = ds.schema.dict_

    @hub.transform(schema=schema_dict, scheduler=scheduler, workers=workers)
    def run_transpose(sample, keys):
        for key in keys:
            if sample[key].dtype == "bool":
                sample[key] = sample[key].astype("uint8")
            sample[key] = Transpose(p=p)(image=sample[key])["image"]
        return sample

    return run_transpose(ds, keys=keys)


if __name__ == "__main__":
    ds = hub.Dataset("activeloop/cifar10_test")[:20]
    transformed_ds = horizonatal_flip(ds, ["image"], workers=2)
    transformed_ds.store("./transformed_cifar")

    ds = hub.Dataset("./transformed_cifar")
    for i, sample in enumerate(ds[:10]):
        sample = sample["image"].compute()
        img = Image.fromarray(sample)
        img.save(f"./img_{i}.png")
