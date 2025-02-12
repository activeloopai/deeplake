import io
import numpy as np
from typing import Callable
from PIL import Image  # type: ignore


def transform(
    sample_in,
    images_tensor: str,
    masks_tensor: str,
    pipeline: Callable,
):
    img = sample_in[images_tensor]
    if isinstance(img, (bytes, bytearray)):
        img = np.array(Image.open(io.BytesIO(img)))
    elif not isinstance(img, np.ndarray):
        img = np.array(img)

    mask = sample_in[masks_tensor]
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    img = img[..., ::-1]  # rgb_to_bgr should be optional
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    shape = img.shape

    pipeline_dict = {
        "img_path": None,
        "seg_map_path": None,
        "label_map": None,  # check if the
        "seg_fields": ["gt_seg_map"],
        "sample_idx": int(sample_in["index"]),  # put the sample idx
        "img": np.ascontiguousarray(img, dtype=np.float32),
        "img_shape": shape[:2],
        "ori_shape": shape[:2],
        "dp_seg_map": np.ascontiguousarray(mask, np.int64),
    }

    return pipeline(pipeline_dict)
