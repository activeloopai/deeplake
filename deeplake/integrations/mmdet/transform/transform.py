from typing import Callable, str, bool

from .transform_class import Transform


def transform(
    sample_in,
    images_tensor: str,
    masks_tensor: str,
    boxes_tensor: str,
    labels_tensor: str,
    pipeline: Callable,
    bbox_info: str,
    poly2mask: bool,
):
    transform_cls = Transform(
        sample_in,
        images_tensor,
        masks_tensor,
        boxes_tensor,
        labels_tensor,
        pipeline,
        bbox_info,
        poly2mask,
    )
    return transform_cls.process()
