import os
from typing import List, Union, Tuple
import numpy as np
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


def horizonatal_flip(
    ds: hub.Dataset,
    keys: Union[List, Tuple] = ["image"],
    p: float = 1.0,
):
    @hub.transform(schema=ds.schema, scheduler="threaded", workers=os.cpu_count())
    def run_flip(sample, keys):
        for key in keys:
            sample[key] = HorizontalFlip(p=p)(image=sample[key])["image"]
        return sample

    return run_flip(ds, keys=keys)


def vertical_flip(
    ds: hub.Dataset, keys: Union[List, Tuple] = ["image"], p: float = 1.0
):
    @hub.transform(schema=ds.schema, scheduler="threaded", workers=os.cpu_count())
    def run_flip(sample, keys):
        for key in keys:
            sample[key] = VerticalFlip(p=p)(image=sample[key])["image"]
        return sample

    return run_flip(ds, keys=keys)


def shift_scale_rotate(
    ds: hub.Dataset,
    keys: Union[List, Tuple] = ["image"],
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
    @hub.transform(schema=ds.schema, scheduler="threaded", workers=os.cpu_count())
    def run_flip(sample, keys):
        for key in keys:
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

    return run_flip(ds, keys=keys)


def blur(
    ds: hub.Dataset,
    keys: Union[List, Tuple] = ["image"],
    p: float = 1.0,
    blur_limit: float = 7,
):
    @hub.transform(schema=ds.schema, scheduler="threaded", workers=os.cpu_count())
    def run_flip(sample, keys):
        for key in keys:
            sample[key] = Blur(p=p, blur_limit=blur_limit)(image=sample[key])["image"]
        return sample

    return run_flip(ds, keys=keys)


def random_brightness_contrast(
    ds: hub.Dataset,
    keys: Union[List, Tuple] = ["image"],
    p: float = 1.0,
    brightness_limit=0.2,
    contrast_limit=0.2,
    brightness_by_max=True,
):
    @hub.transform(schema=ds.schema, scheduler="threaded", workers=os.cpu_count())
    def run_flip(sample, keys):
        for key in keys:
            sample[key] = RandomBrightnessContrast(
                p=p,
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                brightness_by_max=brightness_by_max,
            )(image=sample[key])["image"]
        return sample

    return run_flip(ds, keys=keys)


def gausse_noise(
    ds: hub.Dataset,
    keys: Union[List, Tuple] = ["image"],
    p: float = 1.0,
    var_limit=(10.0, 50.0),
    mean=0,
):
    @hub.transform(schema=ds.schema, scheduler="ray_generator", workers=os.cpu_count())
    def run_flip(sample, keys):
        for key in keys:
            sample[key] = GaussNoise(p=p, var_limit=var_limit, mean=mean)(
                image=sample[key]
            )["image"]
        return sample

    return run_flip(ds, keys=keys)


def transpose(
    ds: hub.Dataset, keys: Union[List, Tuple] = ["image"], p: float = 1.0, **kwargs
):
    @hub.transform(schema=ds.schema, scheduler="threaded", workers=os.cpu_count())
    def run_flip(sample, keys):
        for key in keys:
            sample[key] = Transpose(p=p, **kwargs)(image=sample[key])["image"]
        return sample

    return run_flip(ds, keys=keys)


if __name__ == "__main__":
    ds = hub.Dataset("activeloop/mnist")[:300]
    transformed_ds = gausse_noise(ds, ["image"], mean=0.5)
    transformed_ds.store("./transformed_mnist")

    # ds = hub.Dataset("./transformed_mnist")
    # for i, sample in enumerate(ds[:10]):
    #     sample = sample["image"].compute()
    #     img = Image.fromarray(sample[:, :, 0])
    #     img.save(f"./img_{i}.png")
