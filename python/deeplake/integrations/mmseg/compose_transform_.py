import io
import numpy as np
from typing import Callable, Optional, List
from functools import partial

from deeplake.integrations.mm.exceptions import InvalidImageError, InvalidSegmentError
from deeplake.integrations.mm.upcast_array import upcast_array
from mmcv.utils import build_from_cfg
from mmseg.datasets.builder import PIPELINES  # type: ignore
from mmseg.datasets.pipelines import Compose  # type: ignore


def build_pipeline(steps):
    return Compose(
        [
            build_from_cfg(step, PIPELINES, None)
            for step in steps
            if step["type"] not in {"LoadImageFromFile", "LoadAnnotations"}
        ]
    )


def transform(
    sample_in,
    images_tensor: str,
    masks_tensor: str,
    pipeline: Callable,
):
    try:
        img = upcast_array(sample_in[images_tensor])
    except Exception as e:
        raise InvalidImageError(images_tensor, e)
    if isinstance(img, (bytes, bytearray)):
        img = np.array(Image.open(io.BytesIO(img)))
    elif not isinstance(img, np.ndarray):
        img = np.array(img)

    try:
        mask = sample_in[masks_tensor]
    except Exception as e:
        raise InvalidSegmentMaskError(images_tensor, e)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    img = img[..., ::-1]  # rgb_to_bgr should be optional
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    shape = img.shape

    pipeline_dict = {
        "img": np.ascontiguousarray(img, dtype=np.float32),
        "img_fields": ["img"],
        "filename": None,
        "ori_filename": None,
        "img_shape": shape,
        "ori_shape": shape,
        "gt_semantic_seg": np.ascontiguousarray(mask, np.int64),
        "seg_fields": ["gt_semantic_seg"],
    }

    return pipeline(pipeline_dict)


def compose_transform(
    images_tensor: str,
    masks_tensor: Optional[str],
    pipeline: List,
):
    pipeline = build_pipeline(pipeline)
    return partial(
        transform,
        images_tensor=images_tensor,
        masks_tensor=masks_tensor,
        pipeline=pipeline,
    )
